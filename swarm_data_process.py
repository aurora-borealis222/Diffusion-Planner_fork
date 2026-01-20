import os
import argparse
import json
import zipfile
import pickle

from diffusion_planner.data_process.swarm_data_processor import SwarmDataProcessor


def load_pickles_from_zip(zip_path):
    experiments = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith((".pkl", ".pickle")):
                with zf.open(name) as f:
                    experiments.append(pickle.load(f))
    return experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--future_len", type=int, default=10)
    parser.add_argument("--history_len", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.1)

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    print("Loading pickle archive...")
    experiments = load_pickles_from_zip(args.data_path)
    print(f"Loaded {len(experiments)} experiments")

    processor = SwarmDataProcessor(args)
    processor.work(experiments)

    npz_files = [f for f in os.listdir(args.save_path) if f.endswith(".npz")]

    with open("diffusion_planner_training.json", "w") as f:
        json.dump(sorted(npz_files), f, indent=4)

    print(f"Saved {len(npz_files)} samples")
