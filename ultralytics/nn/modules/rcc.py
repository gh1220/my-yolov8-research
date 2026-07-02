"""RCC modules for YOLOv8.

RCC is composed of a receptive-field attention convolution, coordinate attention, and a final activation.
The module keeps the same input/output shape contract as Conv so it can be dropped into YOLO backbones.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import autopad


class CoordAtt(nn.Module):
    """Coordinate Attention block."""

    def __init__(self, c1: int, c2: int, reduction: int = 32):
        super().__init__()
        mip = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, c2, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, c2, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = torch.cat((x_h, x_w), dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)
        a_h = self.conv_h(y_h).sigmoid()
        a_w = self.conv_w(y_w).sigmoid()
        return x * a_h * a_w


class RFAConv(nn.Module):
    """Receptive-field attention convolution."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p=None, g: int = 1, d: int = 1):
        super().__init__()
        if isinstance(k, tuple):
            k = k[0]
        pad = autopad(k, p, d)
        self.k = k
        self.unfold = nn.Unfold(kernel_size=k, dilation=d, padding=pad, stride=s)
        self.attn = nn.Conv2d(c1, k * k, k, s, pad, dilation=d, bias=True)
        self.proj = nn.Conv2d(c1, c2, 1, 1, 0, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        attn = self.attn(x)
        h, w = attn.shape[-2:]
        patches = self.unfold(x).view(b, c, self.k * self.k, h, w)
        weights = attn.view(b, 1, self.k * self.k, h, w).softmax(dim=2)
        x = (patches * weights).sum(dim=2)
        return self.bn(self.proj(x))


class RCC(nn.Module):
    """Receptive field + Coordinate Attention + RFAConv block.

    The signature mirrors Conv so it can replace Conv layers in YOLOv8 backbones without changing model YAML.
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if isinstance(k, tuple):
            k = k[0]
        self.rfa = RFAConv(c1, c2, k=k, s=s, p=p, g=g, d=d)
        self.ca = CoordAtt(c2, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ca(self.rfa(x)))
