import torch
from collections import defaultdict
from tqdm import tqdm


# -----------------------------
# Метрики (per-sample)
# -----------------------------
def compute_ade(pred, gt):
    pred_xy = pred[..., :2]
    gt_xy = gt[..., :2]

    dist = torch.norm(pred_xy - gt_xy, dim=-1)  # [B, T]
    return dist.mean(dim=-1)  # [B]


def compute_fde(pred, gt):
    pred_xy = pred[:, -1, :2]
    gt_xy = gt[:, -1, :2]

    return torch.norm(pred_xy - gt_xy, dim=-1)  # [B]


# -----------------------------
# Swarm метрики
# -----------------------------
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


# -----------------------------
# Validation
# -----------------------------
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

    for batch in tqdm(data_loader, desc="Validation"):
        *data, exp_idx = batch
        exp_idx = exp_idx.numpy()

        # --------------------------------------------------
        # SUBSET ПО EXPERIMENT
        # --------------------------------------------------
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

        # --------------------------------------------------
        # INPUT
        # --------------------------------------------------
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

        neighbor_future_mask = torch.sum(
            torch.ne(neighbors_future[..., :3], 0),
            dim=-1
        ) == 0

        # neighbor_future_mask = data[11].to(args.device)  # предполагаем, что он есть

        neighbors_valid = ~neighbor_future_mask  # True = валидный

        # --------------------------------------------------
        # heading -> cos/sin
        # --------------------------------------------------
        ego = inputs["ego_current_state"]
        inputs["ego_current_state"] = torch.cat(
            [
                ego[..., :2],
                torch.stack([ego[..., 2].cos(), ego[..., 2].sin()], dim=-1),
            ],
            dim=-1,
        )

        nbr = inputs["neighbor_agents_past"]
        inputs["neighbor_agents_past"] = torch.cat(
            [
                nbr[..., :2],
                torch.stack([nbr[..., 2].cos(), nbr[..., 2].sin()], dim=-1),
            ],
            dim=-1,
        )

        inputs = args.observation_normalizer(inputs)

        # --------------------------------------------------
        # INFERENCE
        # --------------------------------------------------
        _, out = model(inputs)

        pred_all = out["prediction"]  # [B, P, T, 4]

        pred_ego = pred_all[:, 0]
        pred_neighbors = pred_all[:, 1:]

        # GT
        gt_all = torch.cat([ego_future[:, None], neighbors_future], dim=1)

        # mask
        ego_mask = torch.ones_like(ego_future[..., 0], dtype=torch.bool)
        full_mask = torch.cat([ego_mask[:, None], neighbors_valid], dim=1)

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------
        batch_ade = compute_ade(pred_ego, ego_future)
        batch_fde = compute_fde(pred_ego, ego_future)

        batch_swarm_ade = compute_swarm_ade(pred_all, gt_all, full_mask)
        batch_swarm_fde = compute_swarm_fde(pred_all, gt_all, full_mask)

        # --------------------------------------------------
        # SAVE
        # --------------------------------------------------
        for i, e in enumerate(exp_idx):
            exp_metrics[int(e)].append({
                "ade": batch_ade[i].item(),
                "fde": batch_fde[i].item(),
                "swarm_ade": batch_swarm_ade[i].item(),
                "swarm_fde": batch_swarm_fde[i].item(),
            })

    # --------------------------------------------------
    # AGGREGATION
    # --------------------------------------------------
    results = defaultdict(list)

    for e in exp_metrics:
        for k in exp_metrics[e][0].keys():
            val = sum(x[k] for x in exp_metrics[e]) / len(exp_metrics[e])
            results[k].append(val)

    return {k.upper(): sum(v) / len(v) for k, v in results.items()}
