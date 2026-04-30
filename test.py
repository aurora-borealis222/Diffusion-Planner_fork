import torch
from validate import validate_epoch
from torch.utils.data import DataLoader

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.swarm_dataset import SwarmDataset


def run_test(args):

    model = Diffusion_Planner(args).to(args.device)
    model.load_state_dict(torch.load(args.model_path))

    test_set = SwarmDataset(
        data_dir=args.test_set,
        past_neighbor_num=args.agent_num,
        predicted_neighbor_num=args.predicted_neighbor_num,
        data_list=args.test_set_list
    )

    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    metrics = validate_epoch(
        test_loader,
        model,
        args,
        max_experiments=None
    )

    print("TEST RESULT:")
    print(metrics)
