"""CFC modules for YOLOv8.

CFC keeps the same external signature as C2f, but replaces the inner bottleneck convolution path with RCC.
This allows it to be dropped into YOLOv8 backbones without changing tensor shapes or parse_model behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv
from .rcc import RCC


class BottleneckRCC(nn.Module):
    """Bottleneck block with RCC in the second branch."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()
        k1, k2 = k
        if isinstance(k1, tuple):
            k1 = k1[0]
        if isinstance(k2, tuple):
            k2 = k2[0]
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k1, 1)
        self.cv2 = RCC(c_, c2, k2, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class CFC(nn.Module):
    """C2f-style block with Bottleneck-RCC internals."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(BottleneckRCC(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
