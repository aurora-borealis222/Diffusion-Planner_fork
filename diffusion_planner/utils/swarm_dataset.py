import os
import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson


class SwarmDataset(Dataset):
    def __init__(self, data_dir, data_list, past_neighbor_num, predicted_neighbor_num):
        self.data_dir = data_dir
        self.files = openjson(data_list)

        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num

        self.index = []

        for fidx, fname in enumerate(self.files):
            path = os.path.join(self.data_dir, fname)
            with np.load(path) as data:
                S = data["ego_current_state"].shape[0]
            for i in range(S):
                self.index.append((fidx, i))

    def __getitem__(self, idx):
        fidx, i = self.index[idx]
        data = np.load(os.path.join(self.data_dir, self.files[fidx]))

        return (
            data["ego_current_state"][i],
            data["ego_agent_future"][i],
            data["neighbor_agents_past"][i][: self._past_neighbor_num],
            data["neighbor_agents_future"][i][: self._predicted_neighbor_num],
            data["lanes"][i],
            data["lanes_speed_limit"][i],
            data["lanes_has_speed_limit"][i],
            data["route_lanes"][i],
            data["route_lanes_speed_limit"][i],
            data["route_lanes_has_speed_limit"][i],
            data["static_objects"][i],
        )

    def __len__(self):
        return len(self.index)
