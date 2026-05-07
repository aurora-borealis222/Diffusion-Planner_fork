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

    # def to_ego_frame(self, points, anchor):
    #     """
    #     points: (N, 5) -> x, y, vx, vy, heading
    #     anchor: (3,) -> x_ego, y_ego, heading_ego
    #     """
    #
    #     dx = points[:, 0] - anchor[0]
    #     dy = points[:, 1] - anchor[1]
    #
    #     cos = np.cos(-anchor[2])
    #     sin = np.sin(-anchor[2])
    #
    #     # --- position ---
    #     x_new = dx * cos - dy * sin
    #     y_new = dx * sin + dy * cos
    #
    #     # --- velocity (ВАЖНО: без сдвига!) ---
    #     vx = points[:, 2]
    #     vy = points[:, 3]
    #
    #     vx_new = vx * cos - vy * sin
    #     vy_new = vx * sin + vy * cos
    #
    #     # --- heading ---
    #     heading_new = points[:, 4] - anchor[2]
    #
    #     return np.stack([x_new, y_new, vx_new, vy_new, heading_new], axis=-1)

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def to_ego_frame_xyh(self, points, anchor):
        # print("\n--- DEBUG to_ego_frame_xyh ---")
        # print("ego state used (anchor):", anchor)
        # print("difference (points[0] - anchor):", points[0] - anchor)
        # print("first raw point BEFORE transform:", points[0])

        dx = points[:, 0] - anchor[0]
        dy = points[:, 1] - anchor[1]

        cos = np.cos(-anchor[2])
        sin = np.sin(-anchor[2])

        x_new = dx * cos - dy * sin
        y_new = dx * sin + dy * cos

        heading_new = self.normalize_angle(points[:, 2] - anchor[2])

        out = np.stack([x_new, y_new, heading_new], axis=-1)

        # print("first point AFTER transform:", out[0])

        return out


    def to_ego_frame_full(self, points, anchor):
        dx = points[:, 0] - anchor[0]
        dy = points[:, 1] - anchor[1]

        cos = np.cos(-anchor[2])
        sin = np.sin(-anchor[2])

        # position
        x_new = dx * cos - dy * sin
        y_new = dx * sin + dy * cos

        # velocity (без сдвига!)
        vx = points[:, 2]
        vy = points[:, 3]

        vx_new = vx * cos - vy * sin
        vy_new = vx * sin + vy * cos

        # # 👇 ВСТАВЬ ВОТ СЮДА (после расчёта скоростей)
        # print("\n=== DEBUG VELOCITY ===")
        # speed_before = np.sqrt(vx[0] ** 2 + vy[0] ** 2)
        # speed_after = np.sqrt(vx_new[0] ** 2 + vy_new[0] ** 2)
        #
        # print("speed BEFORE:", speed_before)
        # print("speed AFTER:", speed_after)
        # print("SPEED DIFF:", speed_after - speed_before)
        #
        # # 👇 И СЮДА (перед heading)
        # print("\n=== DEBUG HEADING ===")
        # print("heading neighbor:", points[0, 4])
        # print("heading ego:", anchor[2])

        heading_new = self.normalize_angle(points[:, 4] - anchor[2])

        return np.stack([x_new, y_new, vx_new, vy_new, heading_new], axis=-1)


    def normalize(self, data, scale=20.0):
        data[..., 0] /= scale  # x
        data[..., 1] /= scale  # y
        data[..., 2] /= scale  # vx
        data[..., 3] /= scale  # vy
        return data
    # ============================================================
    # Step 3: build one training sample
    # ============================================================
    def _build_sample(self, ego_id, t, trajectories, agent_ids):
        ego_traj = trajectories[ego_id]

        # --- Ego ---
        ego_current_state = self._ego_current_state(ego_traj, t)

        # print("ego_current_state (raw):", ego_current_state)  # ← сюда
        # print("ego position:", ego_current_state[:2])

        ego_future = self._ego_future(ego_traj, t)

        # print("ego_current_state:", ego_current_state)
        # print("raw ego_future[0]:", ego_future[0])

        # --- Neighbors ---
        neighbors_past, neighbors_future = self._neighbors(
            ego_id, t, trajectories, agent_ids
        )

        MAX_RADIUS = 50.0  # можно 50–80

        ego_xy = ego_current_state[:2]

        for i in range(neighbors_past.shape[0]):
            if np.all(neighbors_past[i] == 0):
                continue

            # расстояние до ego (по всем таймстепам)
            dist = np.linalg.norm(
                neighbors_past[i, :, :2] - ego_xy,
                axis=1
            )

            # если ВСЕ точки далеко — выкидываем агента
            if dist.min() > MAX_RADIUS:
                neighbors_past[i] = 0
                neighbors_future[i] = 0

        # print("\n=== RAW CHECK ===")
        #
        # ego_xy = ego_current_state[:2]
        #
        # # берём только валидных соседей (без нулей)
        # mask = ~(np.all(neighbors_past[..., :2] == 0, axis=-1))
        # raw_neighbors = neighbors_past[..., :2][mask]
        #
        # dist = np.linalg.norm(raw_neighbors - ego_xy, axis=1)
        #
        # print("RAW neighbors dist:")
        # print("  max:", dist.max())
        # print("  mean:", dist.mean())
        # print("  min:", dist.min())

        # anchor = ego_future[0].copy()
        ego_future = self.to_ego_frame_xyh(ego_future, ego_current_state)

        # print("ego at origin check (first step):", ego_future[0])
        #
        # print("max abs ego_future xy:", np.abs(ego_future[..., :2]).max())
        # print("mean x ego_future:", ego_future[..., 0].mean())

        # print("neighbor sample BEFORE:", neighbors_past[0, 0])
        # print("position BEFORE:", neighbors_past[0, 0, :2])

        # for i in range(neighbors_past.shape[0]):
        #     neighbors_past[i] = self.to_ego_frame_full(neighbors_past[i], ego_current_state)

        for i in range(neighbors_past.shape[0]):
            if np.all(neighbors_past[i] == 0):
                continue
            neighbors_past[i] = self.to_ego_frame_full(neighbors_past[i], ego_current_state)

        # print("neighbors sample PAST:", neighbors_past[0, 0, :2])

        # print("neighbor sample AFTER:", neighbors_past[0, 0])
        # print("position AFTER:", neighbors_past[0, 0, :2])

        # for i in range(neighbors_future.shape[0]):
        #     neighbors_future[i] = self.to_ego_frame_full(neighbors_future[i], ego_current_state)

        # print("max abs neighbors past:", np.abs(neighbors_past[..., :2]).max())
        # print("mean x neighbors past:", neighbors_past[..., 0].mean())

        for i in range(neighbors_future.shape[0]):
            if np.all(neighbors_future[i] == 0):
                continue
            neighbors_future[i] = self.to_ego_frame_full(neighbors_future[i], ego_current_state)

        # print("neighbors sample FUTURE:", neighbors_future[0, 0, :2])

        valid_mask = ~(np.all(neighbors_future == 0, axis=-1))
        valid_points = neighbors_future[valid_mask]

        if valid_points.size > 0:
            pass
            # print("max abs neighbors FUTURE:", np.abs(valid_points[..., :2]).max())
            # print("mean x neighbors FUTURE:", valid_points[..., 0].mean())
        else:
            pass
            # print("no valid neighbors")


        # print("\n=== SANITY CHECK ===")
        #
        # # ego должен быть в (0,0) на первом timestep
        # print("ego at origin check (first step):", ego_future[0])
        #
        # # любой сосед (если есть)
        # print("neighbors sample FUTURE:", neighbors_future[0, 0, :2])
        # print("neighbors sample PAST:", neighbors_past[0, 0, :2])

        # print("max abs neighbors FUTURE:", np.abs(neighbors_future[..., :2]).max())
        # print("mean x neighbors FUTURE:", neighbors_future[..., 0].mean())

        # ego_current_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)

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
        # return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        s = traj[t]

        # print("heading raw:", s["heading"])

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
        D = 5  # x, y, vx, vy, heading

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
                    s["vel"][0],
                    s["vel"][1],
                    np.deg2rad(s["heading"]),
                ]
                for s in past
            ], dtype=np.float32)

            neighbors_future[k] = np.array(
                [
                    [
                        s["pos"][0],
                        s["pos"][1],
                        s["vel"][0],
                        s["vel"][1],
                        np.deg2rad(s["heading"])
                     ] for s in future
                ],
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

        for k in data:
            if data[k].dtype == np.float32:
                data[k] = data[k].astype(np.float16)

        np.savez_compressed(path, **data)
        print(f"[Saved] {fname}: {len(samples)} samples")
