import torch
import torch.nn as nn

from ..config import EnvConfig
from .base import BaseModel
from .encoder import BaseEncoder


class DQNModel(BaseModel):
    """
    Q-value network for DQN.

    forward() returns Q-values shaped (B, n_actions).
    algo_type is fixed to "QVALUE" so the eval server selects argmax(Q).

    Args:
        encoder:  Any BaseEncoder. The model takes ownership after construction.
        n_actions: Size of the discrete action space (8 for jku.wad).
        env_cfg:  Forwarded to BaseModel for ONNX metadata.
    """

    def __init__(self, encoder: BaseEncoder, n_actions: int, env_cfg: EnvConfig):
        super().__init__(env_cfg, algo_type="QVALUE")
        self.encoder = encoder
        self.q_head = nn.Linear(encoder.out_dim, n_actions)
        nn.init.orthogonal_(self.q_head.weight, gain=1.0)
        nn.init.zeros_(self.q_head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs:    (B, C, H, W) float32 in [0, 1]
        return: q_values (B, n_actions)
        """
        return self.q_head(self.encoder(obs))
