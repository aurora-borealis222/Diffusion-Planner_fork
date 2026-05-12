import torch
import torch.nn as nn

from baselines.transformer_planner import TransformerPlanner


class Transformer_Planner(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.planner = TransformerPlanner(config)

    def forward(self, inputs):

        encoder_outputs, decoder_outputs = self.planner(inputs)

        return {
            **encoder_outputs,
            **decoder_outputs
        }
