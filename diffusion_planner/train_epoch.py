from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from datetime import datetime

from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.swarm_data_augmentation import SwarmStatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func


def train_epoch(data_loader, model, optimizer, args, ema, aug: SwarmStatePerturbation=None):
    epoch_loss = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        history = {
            "total_loss": [],
            "ego_loss": [],
            "neighbor_loss": [],
        }

        for batch in data_epoch:
            '''
            data structure in batch: Tuple(Tensor) 

            ego_current_state,
            ego_future_gt,

            neighbor_agents_past,
            neighbors_future_gt,

            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,

            route_lanes,
            route_lanes_speed_limit,
            route_lanes_has_speed_limit,

            static_objects,

            '''

            # prepare data
            inputs = {
                'ego_current_state': batch[0].to(args.device),

                'neighbor_agents_past': batch[2].to(args.device),

                'lanes': batch[4].to(args.device),
                'lanes_speed_limit': batch[5].to(args.device),
                'lanes_has_speed_limit': batch[6].to(args.device),

                'route_lanes': batch[7].to(args.device),
                'route_lanes_speed_limit': batch[8].to(args.device),
                'route_lanes_has_speed_limit': batch[9].to(args.device),

                'static_objects': batch[10].to(args.device)

            }

            ego_future = batch[1].to(args.device)
            neighbors_future = batch[3].to(args.device)
            # Normalize to ego-centric
            if aug is not None:
                inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

            # heading to cos sin
            ego_future = torch.cat(
            [
                ego_future[..., :2],
                torch.stack(
                    [ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )

            mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
            neighbors_future = torch.cat(
            [
                neighbors_future[..., :2],
                torch.stack(
                    [neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )
            neighbors_future[mask] = 0.

            # ego current: [x, y, heading] -> [x, y, cos, sin]
            ego = inputs["ego_current_state"]
            inputs["ego_current_state"] = torch.cat(
                [
                    ego[..., :2],
                    torch.stack([ego[..., 2].cos(), ego[..., 2].sin()], dim=-1),
                ],
                dim=-1,
            )

            # neighbors current (all past steps)
            # inputs["neighbor_agents_past"] = inputs["neighbor_agents_past"][..., :4]
            neighbors = inputs["neighbor_agents_past"]

            inputs["neighbor_agents_past"] = torch.cat(
                [
                    neighbors[..., :2],  # x,y

                    torch.stack(
                        [
                            neighbors[..., 4].cos(),
                            neighbors[..., 4].sin(),
                        ],
                        dim=-1
                    ),

                    neighbors[..., 2:4],  # vx, vy
                ],
                dim=-1
            )

            # vx = neighbors_past[..., 2]
            # vy = neighbors_past[..., 3]
            #
            # speed = torch.sqrt(vx ** 2 + vy ** 2)
            #
            # valid = torch.sum(torch.ne(neighbors_past[..., :4], 0), dim=-1) > 0
            #
            # speed = speed[valid]
            #
            # print(
            #     "neighbor speed stats:",
            #     "mean =", speed.mean().item(),
            #     "std =", speed.std().item(),
            #     "max =", speed.max().item(),
            # )

            # print("neighbor_agents_past", inputs["neighbor_agents_past"])
            # vx = inputs["neighbor_agents_past"][..., 4]
            # vy = inputs["neighbor_agents_past"][..., 5]

            # print("vx mean", vx.mean())
            # print("vy mean", vy.mean())
            #
            # print("vx std", vx.std())
            # print("vy std", vy.std())

            # valid_mask = torch.sum(
            #     torch.ne(inputs["neighbor_agents_past"], 0),
            #     dim=-1
            # ) > 0

            # vx_valid = vx[valid_mask]
            # vy_valid = vy[valid_mask]

            # print("raw vx mean:", neighbors[..., 2].mean())
            # print("raw vy mean:", neighbors[..., 3].mean())

            # print("VX mean:", vx_valid.mean().item())
            # print("VX std:", vx_valid.std().item())
            #
            # print("VY mean:", vy_valid.mean().item())
            # print("VY std:", vy_valid.std().item())
            #
            # speed = torch.sqrt(vx_valid ** 2 + vy_valid ** 2)
            #
            # print("Speed mean:", speed.mean().item())
            # print("Speed std:", speed.std().item())
            # print("Speed max:", speed.max().item())

            # for k, v in inputs.items():
            #     print(k, v.shape)

            inputs = args.observation_normalizer(inputs)
                  
            # call the mdoel
            optimizer.zero_grad()
            loss = {}

            loss, _ = diffusion_loss_func(
                model,
                inputs,
                ddp.get_model(model, args.ddp).sde.marginal_prob,
                (ego_future, neighbors_future, mask),
                args.state_normalizer,
                loss,
                args.diffusion_model_type
            )

            loss['loss'] = loss['neighbor_prediction_loss'] + args.alpha_planning_loss * loss['ego_planning_loss']

            total_loss = loss['loss'].item()

            # loss backward
            loss['loss'].backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            if ema is not None:
                ema.update(model)

            if args.ddp:
                torch.cuda.synchronize()
            
            data_epoch.set_postfix(loss='{:.4f}'.format(total_loss))
            epoch_loss.append(loss)


    mean_total = np.mean([x["loss"].item() for x in epoch_loss])
    mean_ego = np.mean([x["ego_planning_loss"].item() for x in epoch_loss])
    mean_neighbor = np.mean([x["neighbor_prediction_loss"].item() for x in epoch_loss])

    history["total_loss"].append(mean_total)
    history["ego_loss"].append(mean_ego)
    history["neighbor_loss"].append(mean_neighbor)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_epoch_{timestamp}.npy"

    np.save(filename, history)

    # print(f"Epoch total loss: {mean_total:.4f}")
    # print(f"Epoch ego loss: {mean_ego:.4f}")
    # print(f"Epoch neighbor loss: {mean_neighbor:.4f}")

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")
        
    return epoch_mean_loss, epoch_mean_loss['loss']
