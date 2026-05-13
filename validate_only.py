import torch
from torch.utils.data import DataLoader

import os
import pandas as pd
from datetime import datetime

from torch import optim
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.swarm_dataset import SwarmDataset
from diffusion_planner.validate_epoch import validate_epoch
from diffusion_planner.utils.train_utils import resume_model
from timm.utils import ModelEma

from train_predictor import get_args

from diffusion_planner.utils import ddp
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def main():
    args = get_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)

    model = Diffusion_Planner(args).to(rank if args.device == 'cuda' else args.device)

    params = [{'params': ddp.get_model(model, args.ddp).parameters(), 'lr': args.learning_rate}]

    optimizer = optim.AdamW(params)

    scheduler = CosineAnnealingWarmUpRestarts(optimizer, args.train_epochs, args.warm_up_epoch)

    model_ema = ModelEma(model, decay=0.999, device=args.device)

    model, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(args.resume_model_path,
                                                                                            model,
                                                                                            optimizer, scheduler,
                                                                                            model_ema, args.device)


    print(f"Model loaded from {args.resume_model_path}")

    model = model_ema.ema if model_ema is not None else model
    model.eval()

    val_set = SwarmDataset(
        data_dir=args.val_set,
        data_list=args.val_set_list,
        past_neighbor_num=args.agent_num,
        predicted_neighbor_num=args.predicted_neighbor_num
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Validation samples: {len(val_set)}")

    val_metrics = validate_epoch(
        val_loader,
        model,
        args,
        max_experiments=args.quick_val_experiments
    )

    print("\n===== VALIDATION RESULTS =====")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    save_path = args.resume_model_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"val_metrics_{timestamp}.csv"

    metrics_path = os.path.join(save_path, filename)

    row = {
        **val_metrics
    }

    df = pd.DataFrame([row])

    if not os.path.exists(metrics_path):
        df.to_csv(metrics_path, index=False)
    else:
        df.to_csv(metrics_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
