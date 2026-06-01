import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..config import EnvConfig
from .base import BaseModel
from .encoder import BaseEncoder


class PPOActorCritic(BaseModel):
    """
    Actor-critic model for PPO.

    forward() always returns (logits, value) — during both training and
    inference. The eval server unpacks the tuple correctly. Never add a
    training/eval branch to forward(); ONNX tracing captures one static graph.

    algo_type is fixed to "POLICY" so the eval server samples from the
    Categorical distribution over logits.

    Args:
        encoder:  Any BaseEncoder. The model takes ownership after construction.
        n_actions: Size of the discrete action space (8 for jku.wad).
        env_cfg:  Forwarded to BaseModel for ONNX metadata.
    """

    def __init__(self, encoder: BaseEncoder, n_actions: int, env_cfg: EnvConfig):
        super().__init__(env_cfg, algo_type="POLICY")
        self.encoder = encoder
        self.policy_head = nn.Linear(encoder.out_dim, n_actions)
        self.value_head = nn.Linear(encoder.out_dim, 1)
        # Orthogonal init: small scale on policy head prevents early entropy
        # collapse; standard scale on value head.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        obs:    (B, C, H, W) float32 in [0, 1]
        return: (logits (B, n_actions), value (B, 1))
        """
        features = self.encoder(obs)
        return self.policy_head(features), self.value_head(features)

    # ------------------------------------------------------------------
    # Helpers used only during training — not exported to ONNX
    # ------------------------------------------------------------------

    def act(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy. Used during rollout collection.

        obs: (B, C, H, W) on the model's device.

        Returns:
            action:   (B,) int64  — sampled action indices
            log_prob: (B,)        — log π(a|s)
            entropy:  (B,)        — H[π(·|s)]
            value:    (B, 1)      — V(s) estimate
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions during the PPO update phase.

        obs:     (B, C, H, W)
        actions: (B,) int64

        Returns:
            log_prob: (B,)
            entropy:  (B,)
            value:    (B, 1)
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
