#!/usr/bin/env python3

import os
import json
import random

DATA_DIR = "/home/tatyana/swarm_data_npy"
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

experiments = []

for name in sorted(os.listdir(DATA_DIR)):
    full_path = os.path.join(DATA_DIR, name)

    if not os.path.isdir(full_path):
        continue

    if not name.startswith("experiment_"):
        continue

    has_npy = any(f.endswith(".npy") for f in os.listdir(full_path))
    if has_npy:
        experiments.append(name)

print(f"Found {len(experiments)} valid experiments")

random.seed(SEED)
random.shuffle(experiments)

n = len(experiments)

n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

train = experiments[:n_train]
val   = experiments[n_train:n_train+n_val]
test  = experiments[n_train+n_val:]

with open("train.json", "w") as f:
    json.dump(train, f, indent=2)

with open("val.json", "w") as f:
    json.dump(val, f, indent=2)

with open("test.json", "w") as f:
    json.dump(test, f, indent=2)

print("Split complete:")
print(f"Train: {len(train)}")
print(f"Val:   {len(val)}")
print(f"Test:  {len(test)}")
print(f"Total: {len(train)+len(val)+len(test)}")
