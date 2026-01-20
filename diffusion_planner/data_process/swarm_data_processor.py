import os
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm


class SwarmDataProcessor:
    def __init__(self, args):
        self.save_path = args.save_path

        self.agent_num = args.agent_num
        self.history_len = args.history_len
        self.future_len = args.future_len

        self.dt = args.dt

        os.makedirs(self.save_path, exist_ok=True)

    # ============================================================
    # Main entry
    # ============================================================
    def work(self, experiments):
        for exp_id, experiment in tqdm(enumerate(experiments)):
            fname = f"experiment_{exp_id:04d}.npz"
            path = os.path.join(self.save_path, fname)
            if os.path.exists(path):
                print(f"[Skipped] {fname} already exists")
                continue

            print(f"[Experiment {exp_id}] processing...")
            trajectories = self._build_trajectories(experiment)
            samples = self._build_samples(trajectories)

            if len(samples) == 0:
                print(f"[Experiment {exp_id}] skipped (no valid samples)")
                continue

            self._save_experiment(exp_id, samples)

    # ============================================================
    # Step 1: group states by agent_id
    # ============================================================
    def _build_trajectories(self, experiment):
        from collections import defaultdict
        trajectories = defaultdict(list)

        last_pos = {}

        dt = self.dt  # или dt = 0.1

        for frame in experiment:
            for robot in frame:
                rid = int(robot[0])
                ori = float(robot[1])
                x, y = robot[2]

                pos = np.array([x, y], dtype=np.float32)

                if rid in last_pos:
                    vel = (pos - last_pos[rid]) / dt
                else:
                    vel = np.array([0.0, 0.0], dtype=np.float32)

                trajectories[rid].append({
                    "pos": pos,
                    "vel": vel,
                    "heading": ori,
                })

                last_pos[rid] = pos

        return trajectories

    # ============================================================
    # Step 2: sliding window → samples
    # ============================================================
    def _build_samples(self, trajectories):
        samples = []

        agent_ids = list(trajectories.keys())

        for ego_id in agent_ids:
            traj = trajectories[ego_id]
            T = len(traj)

            if T < self.history_len + self.future_len:
                continue

            for t in range(self.history_len, T - self.future_len):
                sample = self._build_sample(
                    ego_id=ego_id,
                    t=t,
                    trajectories=trajectories,
                    agent_ids=agent_ids,
                )

                if sample is not None:
                    samples.append(sample)

        return samples

    # ============================================================
    # Step 3: build one training sample
    # ============================================================
    def _build_sample(self, ego_id, t, trajectories, agent_ids):
        ego_traj = trajectories[ego_id]

        # --- Ego ---
        ego_current_state = self._ego_current_state(ego_traj, t)
        ego_future = self._ego_future(ego_traj, t)

        # --- Neighbors ---
        neighbors_past, neighbors_future = self._neighbors(
            ego_id, t, trajectories, agent_ids
        )

        # --- Environment (placeholders for now) ---
        MAX_LANES = 1
        MAX_STATIC = 1

        lanes = np.zeros((MAX_LANES, 2), dtype=np.float32)
        lanes_speed_limit = np.zeros((MAX_LANES,), dtype=np.float32)
        lanes_has_speed_limit = np.zeros((MAX_LANES,), dtype=np.bool_)

        route_lanes = np.zeros((MAX_LANES, 2), dtype=np.float32)
        route_lanes_speed_limit = np.zeros((MAX_LANES,), dtype=np.float32)
        route_lanes_has_speed_limit = np.zeros((MAX_LANES,), dtype=np.bool_)

        static_objects = np.zeros((MAX_STATIC, 2), dtype=np.float32)

        return {
            "ego_current_state": ego_current_state,
            "ego_agent_future": ego_future,
            "neighbor_agents_past": neighbors_past,
            "neighbor_agents_future": neighbors_future,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "static_objects": static_objects,
        }

    # ============================================================
    # Ego helpers
    # ============================================================
    def _ego_current_state(self, traj, t):
        s = traj[t]
        return np.array(
            [
                s["pos"][0],
                s["pos"][1],
                #s["vel"][0],
                #s["vel"][1],
                np.deg2rad(s["heading"])
            ],
            dtype=np.float32,
        )

    def _ego_future(self, traj, t):
        future = traj[t + 1 : t + 1 + self.future_len]
        return np.array([[s["pos"][0], s["pos"][1], np.deg2rad(s["heading"])] for s in future], dtype=np.float32)

    # ============================================================
    # Neighbor helpers
    # ============================================================
    def _neighbors(self, ego_id, t, trajectories, agent_ids):
        D = 3  # x, y, heading

        neighbors_past = np.zeros(
            (self.agent_num, self.history_len, D), dtype=np.float32
        )
        neighbors_future = np.zeros(
            (self.agent_num, self.future_len, D), dtype=np.float32
        )

        k = 0
        for aid in agent_ids:
            if aid == ego_id:
                continue

            traj = trajectories[aid]

            if t < self.history_len or t + self.future_len >= len(traj):
                continue

            past = traj[t - self.history_len: t]
            future = traj[t + 1: t + 1 + self.future_len]

            neighbors_past[k] = np.array([
                [
                    s["pos"][0],
                    s["pos"][1],
                    #s["vel"][0],
                    #s["vel"][1],
                    np.deg2rad(s["heading"]),
                ]
                for s in past
            ], dtype=np.float32)

            neighbors_future[k] = np.array(
                [[s["pos"][0], s["pos"][1], np.deg2rad(s["heading"])] for s in future],
                dtype=np.float32
            )

            k += 1
            if k >= self.agent_num:
                break

        return neighbors_past, neighbors_future

    # ============================================================
    # Save
    # ============================================================
    def _save_experiment(self, exp_id, samples):
        fname = f"experiment_{exp_id:04d}.npz"
        path = os.path.join(self.save_path, fname)

        keys = samples[0].keys()
        data = {
            k: np.stack([s[k] for s in samples])
            for k in keys
        }

        np.savez_compressed(path, **data)
        print(f"[Saved] {fname}: {len(samples)} samples")
