"""Spatial Bi-directional Attention for YOLOv8 neck fusion."""

from __future__ import annotations

import torch
import torch.nn as nn


class SBA(nn.Module):
    """Spatial Bi-directional Attention.

    The module is channel-preserving and is designed to be inserted after neck fusion Concat operations.
    It combines horizontal, vertical, and spatial attention so the fused feature is refined in both directions
    while keeping the tensor shape unchanged.
    """

    def __init__(self, c1: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, c1 // reduction)
        self.reduce = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
        )
        self.h_proj = nn.Conv2d(hidden, c1, 1, 1, 0, bias=True)
        self.w_proj = nn.Conv2d(hidden, c1, 1, 1, 0, bias=True)
        self.spatial = nn.Conv2d(2, 1, 7, 1, 3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_reduced = self.reduce(x)

        h_context = x_reduced.mean(dim=3, keepdim=True)
        w_context = x_reduced.mean(dim=2, keepdim=True)

        h_att = self.h_proj(h_context).sigmoid()
        w_att = self.w_proj(w_context).sigmoid()

        avg_map = x.mean(dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        s_att = self.spatial(torch.cat((avg_map, max_map), dim=1)).sigmoid()

        return x * h_att * w_att * s_att
