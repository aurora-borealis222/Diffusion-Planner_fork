import os
import numpy as np
from tqdm import tqdm

SRC_DIR = "/home/tatyana/swarm_data_fixed"
DST_DIR = "/home/tatyana/swarm_data_fixed_npy"

os.makedirs(DST_DIR, exist_ok=True)

npz_files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".npz"))

for fname in tqdm(npz_files, desc="Converting npz -> npy"):
    src_path = os.path.join(SRC_DIR, fname)
    base = os.path.splitext(fname)[0]

    out_dir = os.path.join(DST_DIR, base)
    os.makedirs(out_dir, exist_ok=True)

    with np.load(src_path) as data:
        for key in data.files:
            out_path = os.path.join(out_dir, f"{key}.npy")

            if os.path.exists(out_path):
                continue

            np.save(out_path, data[key])

