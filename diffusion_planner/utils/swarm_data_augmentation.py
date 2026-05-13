import torch
import numpy as np


def vector_transform(vector, transform_mat, bias=None):
    """
    vector: (B, ..., 2)
    transform_mat: (B, 2, 2)
    """

    shape = vector.shape
    B = vector.shape[0]

    if bias is not None:
        nexpand = vector.ndim - 2
        vector = vector - bias.reshape(B, *([1] * nexpand), 2)

    vector = vector.reshape(B, -1, 2)
    vector = vector.permute(0, 2, 1)

    vector = torch.bmm(transform_mat, vector)

    return vector.permute(0, 2, 1).reshape(shape)


def heading_transform(heading, transform_mat):

    B = heading.shape[0]
    shape = heading.shape

    heading = heading.reshape(B, -1)

    transform_mat = transform_mat.reshape(B, 1, 2, 2)

    return torch.atan2(
        torch.cos(heading) * transform_mat[..., 1, 0]
        + torch.sin(heading) * transform_mat[..., 1, 1],

        torch.cos(heading) * transform_mat[..., 0, 0]
        + torch.sin(heading) * transform_mat[..., 0, 1]
    ).reshape(shape)


class SwarmStatePerturbation:

    def __init__(
        self,
        augment_prob=0.5,
        pos_noise=0.5,
        heading_noise=0.25,
        device="cpu"
    ):

        self.augment_prob = augment_prob
        self.device = device

        self.pos_noise = pos_noise
        self.heading_noise = heading_noise

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_transform_matrix_batch(self, heading):

        cos = torch.cos(heading)
        sin = torch.sin(heading)

        return torch.stack([
            torch.stack([ cos, sin], dim=-1),
            torch.stack([-sin, cos], dim=-1),
        ], dim=-2)

    def augment(self, inputs):

        ego = inputs["ego_current_state"].clone()

        B = ego.shape[0]

        aug_flag = (
            torch.rand(B, device=self.device)
            < self.augment_prob
        )

        dx = (
            torch.rand(B, device=self.device) * 2 - 1
        ) * self.pos_noise

        dy = (
            torch.rand(B, device=self.device) * 2 - 1
        ) * self.pos_noise

        dheading = (
            torch.rand(B, device=self.device) * 2 - 1
        ) * self.heading_noise

        ego[aug_flag, 0] += dx[aug_flag]
        ego[aug_flag, 1] += dy[aug_flag]

        ego[aug_flag, 2] += dheading[aug_flag]

        ego[:, 2] = self.normalize_angle(ego[:, 2])

        return ego

    def centric_transform(
        self,
        inputs,
        ego_future,
        neighbors_future
    ):

        ego = inputs["ego_current_state"]

        center_xy = ego[:, :2]
        heading = ego[:, 2]

        transform_matrix = self.get_transform_matrix_batch(heading)

        ego_future[..., :2] = vector_transform(
            ego_future[..., :2],
            transform_matrix,
            center_xy
        )

        ego_future[..., 2] = heading_transform(
            ego_future[..., 2],
            transform_matrix
        )

        mask = torch.sum(
            torch.ne(inputs["neighbor_agents_past"], 0),
            dim=-1
        ) == 0

        inputs["neighbor_agents_past"][..., :2] = vector_transform(
            inputs["neighbor_agents_past"][..., :2],
            transform_matrix,
            center_xy
        )

        inputs["neighbor_agents_past"][..., 2:4] = vector_transform(
            inputs["neighbor_agents_past"][..., 2:4],
            transform_matrix
        )

        inputs["neighbor_agents_past"][..., 4] = heading_transform(
            inputs["neighbor_agents_past"][..., 4],
            transform_matrix
        )

        inputs["neighbor_agents_past"][mask] = 0.

        mask = torch.all(
            neighbors_future == 0,
            dim=-1
        )

        neighbors_future[..., :2] = vector_transform(
            neighbors_future[..., :2],
            transform_matrix,
            center_xy
        )

        neighbors_future[..., 2] = heading_transform(
            neighbors_future[..., 2],
            transform_matrix
        )

        neighbors_future[mask] = 0.

        inputs["ego_current_state"][:, 0] = 0.
        inputs["ego_current_state"][:, 1] = 0.
        inputs["ego_current_state"][:, 2] = 0.

        return inputs, ego_future, neighbors_future

    def __call__(self, inputs, ego_future, neighbors_future):

        inputs["ego_current_state"] = self.augment(inputs)

        return self.centric_transform(
            inputs,
            ego_future,
            neighbors_future
        )
