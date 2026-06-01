from abc import abstractmethod

import torch
import torch.nn as nn

from ..config import EnvConfig

# Type alias for what forward() may return
Logits = torch.Tensor
Value = torch.Tensor


class BaseModel(nn.Module):
    """
    Base class for all submitted models.

    Subclasses must implement forward(). The output is used directly during
    ONNX export and by the eval server, so the signature must match:
        - input:  (B, C, H, W) float32 tensor, normalized to [0, 1]
        - output: logits (B, n_actions)  — or (logits, value) for actor-critic

    The onnx_config property produces the JSON metadata the grading server reads.
    """

    def __init__(self, env_cfg: EnvConfig, algo_type: str = "POLICY"):
        super().__init__()
        self.env_cfg = env_cfg
        self.algo_type = algo_type

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> Logits | tuple[Logits, Value]:
        ...

    @property
    def onnx_config(self) -> dict:
        return {
            "screen_format": self.env_cfg.screen_format,
            "n_stack_frames": self.env_cfg.n_stack_frames,
            "extra_state": self.env_cfg.extra_state,
            "hud": self.env_cfg.hud,
            "crosshair": self.env_cfg.crosshair,
            "algo_type": self.algo_type,
        }
