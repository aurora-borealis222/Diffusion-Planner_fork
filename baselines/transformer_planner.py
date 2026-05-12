import torch
import torch.nn as nn

from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.utils.normalizer import StateNormalizer


class TransformerTrajectoryDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim=192,
        future_len=10,
        num_layers=4,
        num_heads=8,
        state_normalizer=None
    ):
        super().__init__()

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_len * 4)
        )

        self.future_len = future_len
        self._state_normalizer: StateNormalizer = state_normalizer

    def forward(self, encoding, mask, denormalize=False):

        x = self.transformer(
            encoding,
            src_key_padding_mask=mask
        )

        x = self.head(x)

        B, P, _ = x.shape

        x = x.view(B, P, self.future_len, 4)

        if denormalize:
            x = self._state_normalizer.inverse(x)

        return {
            "prediction": x
        }


class TransformerPlanner(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)

        self.decoder = TransformerTrajectoryDecoder(
            hidden_dim=config.hidden_dim,
            future_len=config.future_len,
            num_layers=4,
            num_heads=config.num_heads,
            state_normalizer=config.state_normalizer
        )

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        neighbors = inputs["neighbor_agents_past"]

        neighbor_mask = (
            torch.sum(torch.ne(neighbors[..., :4], 0), dim=-1) == 0
        )

        agent_mask = torch.sum(~neighbor_mask, dim=-1) == 0

        pred = self.decoder(
            encoder_outputs["encoding"],
            mask=agent_mask,
            denormalize=not self.training
        )

        return encoder_outputs, pred
