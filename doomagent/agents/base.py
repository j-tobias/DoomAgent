from abc import ABC, abstractmethod
from pathlib import Path

import torch

from ..config import TrainConfig
from ..models.base import BaseModel


class BaseAgent(ABC):
    """
    Shared interface for all RL agents.

    Subclasses own the training loop (collect + update). The base class
    provides checkpointing and ONNX export so every agent gets them for free.
    """

    def __init__(self, model: BaseModel, cfg: TrainConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.step = 0  # global environment step counter

    @abstractmethod
    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> int:
        """Inference-time action selection. obs: (C, H, W) tensor, already on CPU."""
        ...

    @abstractmethod
    def update(self) -> dict[str, float]:
        """
        One gradient update step.
        Returns a dict of scalar metrics (loss, entropy, etc.) for logging.
        """
        ...

    def save(self, path: str | Path) -> None:
        from ..utils.checkpoint import save_checkpoint
        save_checkpoint(self, Path(path))

    def load(self, path: str | Path) -> None:
        from ..utils.checkpoint import load_checkpoint
        load_checkpoint(self, Path(path))

    def export_onnx(self, path: str | Path, obs_shape: tuple) -> None:
        """Export the model to ONNX with eval-server metadata attached."""
        from ..utils.export import export_onnx
        export_onnx(self.model, Path(path), obs_shape)
