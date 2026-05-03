import torch
from torch.utils.data import DataLoader

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

    print("Normalizer keys:", args.observation_normalizer._normalization_dict.keys())

    # ------------------------
    # MODEL
    # ------------------------
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)

    model = Diffusion_Planner(args).to(rank if args.device == 'cuda' else args.device)

    params = [{'params': ddp.get_model(model, args.ddp).parameters(), 'lr': args.learning_rate}]

    optimizer = optim.AdamW(params)

    train_epochs = 1
    warm_up_epoch = 0
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, warm_up_epoch)

    model_ema = ModelEma(model, decay=0.999, device=args.device)

    model, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(args.resume_model_path,
                                                                                            model,
                                                                                            optimizer, scheduler,
                                                                                            model_ema, args.device)

    # model, optimizer, scheduler, epoch, _, model_ema = resume_model(
    #     args.resume_model_path,
    #     model,
    #     optimizer,
    #     scheduler,
    #     model_ema,
    #     args.device
    # )

    print(f"Model loaded from {args.resume_model_path}")

    # 👉 используем EMA если есть
    # model = model_ema.ema if model_ema is not None else model
    model.eval()

    # ------------------------
    # DATA
    # ------------------------
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

    # ------------------------
    # VALIDATION
    # ------------------------
    metrics = validate_epoch(
        val_loader,
        model,
        args,
        max_experiments=args.quick_val_experiments
    )

    print(f"[Quick Val] ADE={metrics['ADE']:.4f}")

    print("\n===== VALIDATION RESULTS =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
