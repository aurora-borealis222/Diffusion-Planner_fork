import os
import numpy as np
import json
from torch.utils.data import Dataset


ARRAY_KEYS = [
    "ego_current_state",
    "ego_agent_future",
    "neighbor_agents_past",
    "neighbor_agents_future",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
    "route_lanes",
    "route_lanes_speed_limit",
    "route_lanes_has_speed_limit",
    "static_objects",
]


class SwarmDataset(Dataset):
    """
    Dataset для Diffusion Planner, безопасный для mmap + DataLoader.

    - mmap'ит ТОЛЬКО один эксперимент на worker
    - не превышает лимит открытых файлов
    - подходит для большого числа экспериментов
    """

    def __init__(
        self,
        data_dir: str,
        past_neighbor_num: int,
        predicted_neighbor_num: int,
        data_list: str = None
    ):
        self.data_dir = data_dir
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num

        if data_list is None:
            self.exp_dirs = sorted(
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            )
        else:
            with open(data_list, "r") as f:
                self.exp_dirs = json.load(f)

        if len(self.exp_dirs) == 0:
            raise RuntimeError(f"No experiments found in {data_dir}")

        self.index = []

        for exp_idx, exp_name in enumerate(self.exp_dirs):
            ego_path = os.path.join(
                data_dir, exp_name, "ego_current_state.npy"
            )
            ego = np.load(ego_path, mmap_mode="r")
            num_frames = ego.shape[0]

            for i in range(num_frames):
                self.index.append((exp_idx, i))

        self._current_exp_idx = None
        self._arrays = None


    def _load_experiment(self, exp_idx: int):
        """
        Загружает mmap для одного эксперимента.
        Если уже загружен — ничего не делает.
        """
        if self._current_exp_idx == exp_idx:
            return

        exp_path = os.path.join(self.data_dir, self.exp_dirs[exp_idx])

        self._arrays = {
            k: np.load(
                os.path.join(exp_path, f"{k}.npy"),
                mmap_mode="r",
            )
            for k in ARRAY_KEYS
        }

        self._current_exp_idx = exp_idx

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        exp_idx, frame_idx = self.index[idx]

        self._load_experiment(exp_idx)
        d = self._arrays
        
        assert (
            d["ego_current_state"].shape[0]
            == d["ego_agent_future"].shape[0]
            == d["neighbor_agents_past"].shape[0]
            == d["neighbor_agents_future"].shape[0]
        )

        return (
            np.array(d["ego_current_state"][frame_idx], copy=True),
            np.array(d["ego_agent_future"][frame_idx], copy=True),
            np.array(d["neighbor_agents_past"][frame_idx][: self._past_neighbor_num], copy=True),
            np.array(d["neighbor_agents_future"][frame_idx][: self._predicted_neighbor_num], copy=True),
            np.array(d["lanes"][frame_idx], copy=True),
            np.array(d["lanes_speed_limit"][frame_idx], copy=True),
            np.array(d["lanes_has_speed_limit"][frame_idx], copy=True),
            np.array(d["route_lanes"][frame_idx], copy=True),
            np.array(d["route_lanes_speed_limit"][frame_idx], copy=True),
            np.array(d["route_lanes_has_speed_limit"][frame_idx], copy=True),
            np.array(d["static_objects"][frame_idx], copy=True),
            exp_idx
        )

