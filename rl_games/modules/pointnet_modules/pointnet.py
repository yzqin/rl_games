# Code source from Jiayuan Gu: https://github.com/Jiayuan-Gu/torkit3d
import torch
import torch.nn as nn

from ..common.mlp import mlp1d_bn_relu, mlp_bn_relu, mlp_relu, mlp1d_relu

__all__ = ["PointNet"]


class PointNet(nn.Module):
    """PointNet for classification.
    Notes:
        1. The original implementation includes dropout for global MLPs.
        2. The original implementation decays the BN momentum.
    """

    def __init__(
            self,
            in_channels=3,
            local_channels=(64, 64, 64, 128, 1024),
            global_channels=(512, 256),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = (local_channels + global_channels)[-1]

        self.mlp_local = mlp1d_bn_relu(in_channels, local_channels)
        self.mlp_global = mlp_bn_relu(local_channels[-1], global_channels)

        self.reset_parameters()

    def forward(self, points, points_feature=None, points_mask=None) -> dict:
        # points: [B, 3, N]; points_feature: [B, C, N], points_mask: [B, N]
        if points_feature is not None:
            input_feature = torch.cat([points, points_feature], dim=1)
        else:
            input_feature = points

        local_feature = self.mlp_local(input_feature)
        if points_mask is not None:
            local_feature = torch.where(
                points_mask.unsqueeze(1), local_feature, torch.zeros_like(local_feature)
            )
        global_feature, max_indices = torch.max(local_feature, 2)
        output_feature = self.mlp_global(global_feature)

        return {"feature": output_feature, "max_indices": max_indices}

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = 0.01
