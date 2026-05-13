import numpy as np
import glob

sum_vx = 0.0
sum_vy = 0.0

sum_vx2 = 0.0
sum_vy2 = 0.0

count = 0


files = glob.glob("/home/tatyana/swarm_data_fixed/*.npz")

print("num files:", len(files))
print(files[:5])


for f in files:
    data = np.load(f)

    arr = data["neighbor_agents_past"]  # [A, T, D]

    vx = arr[..., 2]
    vy = arr[..., 3]

    mask = np.any(arr != 0, axis=-1)

    vx_valid = vx[mask]
    vy_valid = vy[mask]

    sum_vx += vx_valid.sum()
    sum_vy += vy_valid.sum()

    sum_vx2 += (vx_valid ** 2).sum()
    sum_vy2 += (vy_valid ** 2).sum()

    count += vx_valid.size


vx_mean = sum_vx / count
vy_mean = sum_vy / count

vx_std = np.sqrt(sum_vx2 / count - vx_mean ** 2)
vy_std = np.sqrt(sum_vy2 / count - vy_mean ** 2)

print("VX mean:", vx_mean)
print("VX std:", vx_std)
print("VY mean:", vy_mean)
print("VY std:", vy_std)
