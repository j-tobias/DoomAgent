from abc import ABC, abstractmethod
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEncoder(nn.Module, ABC):
    """
    Contract for all visual encoders.

    Subclasses must set `self.out_dim: int` in __init__ so that model heads
    can query the feature vector width at construction time — no dummy forward
    pass required.

    forward() maps (B, C, H, W) float32 in [0, 1] → (B, out_dim) features.
    """
    out_dim: int  # set by subclass __init__

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, C, H, W) → features: (B, out_dim)"""
        ...


# ---------------------------------------------------------------------------
# Building blocks (adapted from jku.wad/agents/utils.py)
# Copied rather than imported: jku.wad/agents/ has no __init__.py so
# cross-.pth imports are fragile in script contexts.
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """Strided convolution downsampling with SiLU activation. space=2 for images."""

    def __init__(self, space: int, dim: int, downsample: int = 2):
        super().__init__()
        Conv = nn.Conv2d if space == 2 else nn.Conv3d
        stride = downsample if space == 2 else (1, downsample, downsample)
        self.conv = Conv(dim, dim, downsample, stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv(x))


class ResidualBlock(nn.Module):
    """Two-branch residual block with configurable depth and activation."""

    def __init__(
        self,
        space: int,
        dim: int,
        act_fn: Type[nn.Module] = nn.SiLU,
        depth: int = 2,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        Conv = nn.Conv2d if space == 2 else nn.Conv3d
        convs = []
        for d in range(depth):
            branch = nn.Sequential(
                Conv(dim, dim, kernel_size=kernel_size, padding=padding),
                act_fn(),
                Conv(dim, dim, kernel_size=kernel_size, padding=padding),
            )
            convs.append(branch)
            if d < depth - 1:
                convs.append(act_fn())
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for conv in self.convs:
            x = conv(x) + residual
            residual = x
        return x


# ---------------------------------------------------------------------------
# IMPALAEncoder — residual CNN for 128×128 input
# ---------------------------------------------------------------------------

class IMPALAEncoder(BaseEncoder):
    """
    CNN from the IMPALA paper (Espeholt et al. 2018), adapted for 128×128 input.

    Three convolutional stacks, each followed by residual blocks. The skip
    connections stabilise gradients and prevent the value-function collapse
    seen with plain CNNs under sparse rewards.

    Spatial resolution trace (128×128 input):
        Conv(in_ch → 16, 3×3) + MaxPool(3, 2, 1)  →  64×64
        2 × ResBlock(16)
        Conv(16 → 32, 3×3)   + MaxPool(3, 2, 1)  →  32×32
        2 × ResBlock(32)
        Conv(32 → 32, 3×3)   + MaxPool(3, 2, 1)  →  16×16
        2 × ResBlock(32)
        ReLU → Flatten → Linear(8192, out_dim)

    Args:
        in_channels: input channel count (3 RGB, 4 with labels, etc.)
        out_dim:     feature vector size (default 256 — IMPALA is efficient
                     enough that 256 matches NatureCNN's 512 in practice).
    """

    # Stack config: (out_channels, n_residual_blocks)
    _STACK = [(16, 2), (32, 2), (32, 2)]

    def __init__(self, in_channels: int, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim

        stacks = []
        ch = in_channels
        for out_ch, n_blocks in self._STACK:
            stacks.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1))
            stacks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            for _ in range(n_blocks):
                stacks.append(ResidualBlock(space=2, dim=out_ch, act_fn=nn.ReLU, depth=1))
            ch = out_ch

        self.conv = nn.Sequential(*stacks)
        self.act = nn.ReLU()

        with torch.no_grad():
            flat_dim = self.act(self.conv(torch.zeros(1, in_channels, 128, 128))).flatten(1).shape[1]
        self.linear = nn.Linear(flat_dim, out_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(self.conv(obs)).flatten(1))


# ---------------------------------------------------------------------------
# NatureCNN — NatureDQN-style encoder for 128×128 input
# ---------------------------------------------------------------------------

class NatureCNN(BaseEncoder):
    """
    Three-layer CNN from the NatureDQN paper, sized for 128×128 input.

    Spatial resolution trace (128×128 input):
        Conv(in_ch → 32, k=8, s=4)  →  31×31
        Conv(32    → 64, k=4, s=2)  →  14×14
        Conv(64    → 64, k=3, s=1)  →  12×12
        Flatten → Linear(9216, out_dim)

    Args:
        in_channels: number of input channels (3 for RGB, 1 for grayscale,
                     more if extra buffers or frame stacking are enabled).
        out_dim:     size of the output feature vector (default 512).
    """

    def __init__(self, in_channels: int, out_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened size once; avoids magic numbers
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128)
            flat_dim = self.cnn(dummy).shape[1]
        self.linear = nn.Linear(flat_dim, out_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs))
