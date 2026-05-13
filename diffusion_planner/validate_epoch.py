import torch
from collections import defaultdict
from tqdm import tqdm

from diffusion_planner.utils.swarm_data_augmentation import SwarmStatePerturbation


def compute_ade(pred, gt):
    pred_xy = pred[..., :2]
    gt_xy = gt[..., :2]

    dist = torch.norm(pred_xy - gt_xy, dim=-1)  # [B, T]
    return dist.mean(dim=-1)  # [B]


def compute_fde(pred, gt):
    pred_xy = pred[:, -1, :2]
    gt_xy = gt[:, -1, :2]

    return torch.norm(pred_xy - gt_xy, dim=-1)  # [B]


def compute_swarm_ade(pred, gt, mask):
    valid_mask = ~mask

    pred_xy = pred[..., :2]
    gt_xy = gt[..., :2]

    dist = torch.norm(pred_xy - gt_xy, dim=-1)  # [B, P, T]

    dist = dist * valid_mask
    denom = valid_mask.sum(dim=(1, 2)) + 1e-6

    return dist.sum(dim=(1, 2)) / denom


def compute_swarm_fde(pred, gt, mask):
    valid_mask = ~mask

    pred_xy = pred[..., :2]
    gt_xy = gt[..., :2]

    final_dist = torch.norm(pred_xy[:, :, -1] - gt_xy[:, :, -1], dim=-1)

    final_mask = valid_mask[:, :, -1]

    final_dist = final_dist * final_mask
    denom = final_mask.sum(dim=1) + 1e-6

    return final_dist.sum(dim=1) / denom


def constant_velocity_predict(
    neighbors_past,
    future_len=10,
    dt=1.0
):

    last_state = neighbors_past[:, :, -1]

    x0 = last_state[..., 0]
    y0 = last_state[..., 1]

    vx = last_state[..., 4]
    vy = last_state[..., 5]

    heading = torch.atan2(vy, vx)

    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)

    preds = []

    for t in range(1, future_len + 1):

        xt = x0 + vx * t * dt
        yt = y0 + vy * t * dt

        pred_t = torch.stack(
            [
                xt,
                yt,
                cos_h,
                sin_h,
            ],
            dim=-1
        )

        preds.append(pred_t)

    return torch.stack(preds, dim=2)


def constant_acceleration_predict(
    neighbors_past,
    future_len=10,
    dt=1.0
):
    """
    neighbors_past:
        [B, P, H, 6]

    features:
        x, y, cos, sin, vx, vy

    returns:
        [B, P, T, 4]
    """

    B, P, H, D = neighbors_past.shape

    device = neighbors_past.device

    last_pos = neighbors_past[:, :, -1, :2]     # [B,P,2]

    last_cos = neighbors_past[:, :, -1, 2]
    last_sin = neighbors_past[:, :, -1, 3]

    last_vel = neighbors_past[:, :, -1, 4:6]    # [B,P,2]

    prev_vel = neighbors_past[:, :, -2, 4:6]

    accel = (last_vel - prev_vel) / dt

    pred = torch.zeros(
        B,
        P,
        future_len,
        4,
        device=device
    )

    for t in range(future_len):

        tau = (t + 1) * dt

        pos_t = (
            last_pos
            + last_vel * tau
            + 0.5 * accel * (tau ** 2)
        )

        pred[:, :, t, :2] = pos_t

        pred[:, :, t, 2] = last_cos
        pred[:, :, t, 3] = last_sin


    invalid_mask = torch.sum(
        torch.ne(neighbors_past, 0),
        dim=-1
    ) == 0

    invalid_mask = invalid_mask[:, :, -1]

    pred[invalid_mask] = 0.

    return pred


@torch.no_grad()
def validate_epoch(
    data_loader,
    model,
    args,
    max_experiments=None,
):
    model.eval()

    exp_metrics = defaultdict(list)
    used_experiments = set()

    val_transform = SwarmStatePerturbation(
        augment_prob=0.0,
        device=args.device
    )

    for batch in tqdm(data_loader, desc="Validation"):
        *data, exp_idx = batch
        exp_idx = exp_idx.numpy()

        if max_experiments is not None:

            batch_exps = set(int(e) for e in exp_idx)
            new_exps = [e for e in batch_exps if e not in used_experiments]

            if len(used_experiments) >= max_experiments:
                mask = [int(e) in used_experiments for e in exp_idx]

                if not any(mask):
                    continue

                data = [d[mask] for d in data]
                exp_idx = exp_idx[mask]

            else:
                for e in new_exps:
                    if len(used_experiments) < max_experiments:
                        used_experiments.add(e)

        inputs = {
            'ego_current_state': data[0].to(args.device),
            'neighbor_agents_past': data[2].to(args.device),

            'lanes': data[4].to(args.device),
            'lanes_speed_limit': data[5].to(args.device),
            'lanes_has_speed_limit': data[6].to(args.device),

            'route_lanes': data[7].to(args.device),
            'route_lanes_speed_limit': data[8].to(args.device),
            'route_lanes_has_speed_limit': data[9].to(args.device),

            'static_objects': data[10].to(args.device),
        }

        ego_future = data[1].to(args.device)
        neighbors_future = data[3].to(args.device)

        inputs, ego_future, neighbors_future = val_transform.centric_transform(
            inputs,
            ego_future,
            neighbors_future
        )

        neighbor_future_mask = torch.sum(
            torch.ne(neighbors_future[..., :3], 0),
            dim=-1
        ) == 0

        # heading -> cos/sin
        ego_future = torch.cat(
            [
                ego_future[..., :2],
                torch.stack([ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1),
            ],
            dim=-1,
        )

        neighbors_future = torch.cat(
            [
                neighbors_future[..., :2],
                torch.stack([neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1),
            ],
            dim=-1,
        )

        neighbors_future[neighbor_future_mask] = 0.

        ego = inputs["ego_current_state"]
        inputs["ego_current_state"] = torch.cat(
            [
                ego[..., :2],
                torch.stack([ego[..., 2].cos(), ego[..., 2].sin()], dim=-1),
            ],
            dim=-1,
        )

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

        inputs = args.observation_normalizer(inputs)
        _, out = model(inputs)

        pred_all = out["prediction"]  # [B, P, T, 4]


        pred_ego = pred_all[:, 0]
        pred_neighbors = pred_all[:, 1:]

        gt_all = torch.cat([ego_future[:, None], neighbors_future], dim=1)

        ego_mask = torch.zeros_like(ego_future[..., 0], dtype=torch.bool)  # ego всегда валиден

        full_mask = torch.cat(
            [ego_mask[:, None], neighbor_future_mask],
            dim=1
        )

        batch_ade = compute_ade(pred_ego, ego_future)
        batch_fde = compute_fde(pred_ego, ego_future)

        batch_swarm_ade = compute_swarm_ade(pred_all, gt_all, full_mask)
        batch_swarm_fde = compute_swarm_fde(pred_all, gt_all, full_mask)


        cv_pred_neighbors = constant_velocity_predict(
            inputs["neighbor_agents_past"],
            future_len=neighbors_future.shape[2]
        )

        B = ego_future.shape[0]
        T = ego_future.shape[1]

        cv_pred_ego = torch.zeros(
            B,
            T,
            4,
            device=args.device
        )

        cv_pred_ego[..., 2] = 1.0

        cv_pred_all = torch.cat(
            [
                cv_pred_ego[:, None],
                cv_pred_neighbors
            ],
            dim=1
        )

        cv_batch_ade = compute_ade(
            cv_pred_ego,
            ego_future
        )

        cv_batch_fde = compute_fde(
            cv_pred_ego,
            ego_future
        )

        cv_batch_swarm_ade = compute_swarm_ade(
            cv_pred_all,
            gt_all,
            full_mask
        )

        cv_batch_swarm_fde = compute_swarm_fde(
            cv_pred_all,
            gt_all,
            full_mask
        )

        ca_pred_neighbors = constant_acceleration_predict(
            inputs["neighbor_agents_past"],
            future_len=neighbors_future.shape[2]
        )

        ca_pred_ego = torch.zeros(
            B,
            T,
            4,
            device=args.device
        )

        ca_pred_ego[..., 2] = 1.0

        ca_pred_all = torch.cat(
            [
                ca_pred_ego[:, None],
                ca_pred_neighbors
            ],
            dim=1
        )

        ca_batch_ade = compute_ade(
            ca_pred_ego,
            ego_future
        )

        ca_batch_fde = compute_fde(
            ca_pred_ego,
            ego_future
        )

        ca_batch_swarm_ade = compute_swarm_ade(
            ca_pred_all,
            gt_all,
            full_mask
        )

        ca_batch_swarm_fde = compute_swarm_fde(
            ca_pred_all,
            gt_all,
            full_mask
        )

        scene_scale = gt_all[..., :2].abs().mean()

        rel_ade = batch_swarm_ade.mean() / scene_scale
        cv_rel_ade = cv_batch_swarm_ade.mean() / scene_scale
        ca_rel_ade = ca_batch_swarm_ade.mean() / scene_scale

        print("Scene scale:", scene_scale.item())

        print("Model Relative ADE:", rel_ade.item())
        print("CV Relative ADE:", cv_rel_ade.item())
        print("CA Relative ADE:", ca_rel_ade.item())


        for i, e in enumerate(exp_idx):
            exp_metrics[int(e)].append({
                "ade": batch_ade[i].item(),
                "fde": batch_fde[i].item(),
                "swarm_ade": batch_swarm_ade[i].item(),
                "swarm_fde": batch_swarm_fde[i].item(),

                "cv_ade": cv_batch_ade[i].item(),
                "cv_fde": cv_batch_fde[i].item(),
                "cv_swarm_ade": cv_batch_swarm_ade[i].item(),
                "cv_swarm_fde": cv_batch_swarm_fde[i].item(),

                "ca_ade": ca_batch_ade[i].item(),
                "ca_fde": ca_batch_fde[i].item(),
                "ca_swarm_ade": ca_batch_swarm_ade[i].item(),
                "ca_swarm_fde": ca_batch_swarm_fde[i].item(),
            })


    results = defaultdict(list)

    for e in exp_metrics:
        for k in exp_metrics[e][0].keys():
            val = sum(x[k] for x in exp_metrics[e]) / len(exp_metrics[e])
            results[k].append(val)

    return {k.upper(): sum(v) / len(v) for k, v in results.items()}
