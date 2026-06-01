import copy
import random

import torch
import torch.optim as optim

from ..buffers.replay import ReplayBuffer
from ..config import DQNConfig
from ..models.dqn import DQNModel
from ..utils.logger import Logger
from .base import BaseAgent


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with epsilon-greedy exploration and EMA target network.

    Training flow:
        agent.setup(obs_shape)          # allocate ReplayBuffer
        agent.train(env, logger)
            └── episode loop:
                    obs = env.reset()
                    step loop:
                        action = epsilon_greedy(obs)
                        buffer.add(...)
                        if len(buffer) >= warmup_steps:
                            update()
                            if grad_steps % target_update_freq == 0:
                                _sync_target()
                    decay_epsilon()     # once per episode
    """

    def __init__(
        self,
        model: DQNModel,
        cfg: DQNConfig,
        device: torch.device,
    ):
        super().__init__(model, cfg, device)
        self.cfg: DQNConfig = cfg
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        # Target network — frozen copy of the online network
        self.target_model = copy.deepcopy(model).to(device)
        for p in self.target_model.parameters():
            p.requires_grad_(False)
        self._buffer: ReplayBuffer | None = None
        self._epsilon: float = cfg.epsilon_start
        self._grad_steps: int = 0   # counts gradient updates for target sync

    def setup(self, obs_shape: tuple) -> None:
        """
        Allocate the ReplayBuffer. Same lazy-init pattern as PPOAgent.
        """
        self._buffer = ReplayBuffer(
            capacity=self.cfg.buffer_size,
            obs_shape=obs_shape,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> int:
        """
        Greedy evaluation action — argmax Q-value, no exploration.
        obs: (C, H, W) CPU tensor.

        During training use epsilon_greedy() instead.
        """
        obs = obs.unsqueeze(0).to(self.device, dtype=torch.float32)
        return self.model(obs).argmax(-1).cpu().item()

    def update(self) -> dict[str, float]:
        """
        One gradient step: sample a minibatch, compute Bellman loss against
        the frozen target network, apply one Adam step.

        Increments self._grad_steps. Caller is responsible for invoking
        _sync_target() at the appropriate frequency.

        Returns a dict with keys: loss, epsilon, buffer_size.

        Raises RuntimeError if setup() has not been called or buffer has
        fewer than cfg.batch_size transitions.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training helpers — real implementations
    # ------------------------------------------------------------------

    @torch.no_grad()
    def epsilon_greedy(self, obs: torch.Tensor) -> int:
        """
        Epsilon-greedy action for exploration during training.
        obs: (C, H, W) CPU tensor.

        With probability self._epsilon returns a uniformly random action.
        Otherwise delegates to select_action() (greedy).
        """
        if random.random() < self._epsilon:
            return random.randrange(self.model.q_head.out_features)
        return self.select_action(obs)

    def decay_epsilon(self) -> None:
        """
        Multiplicative epsilon decay. Call once per episode end.
        Clips from below at cfg.epsilon_end.
        """
        self._epsilon = max(
            self.cfg.epsilon_end,
            self._epsilon * self.cfg.epsilon_decay,
        )

    # ------------------------------------------------------------------
    # Training helpers — stubs
    # ------------------------------------------------------------------

    def _sync_target(self) -> None:
        """
        Copy online network weights into the target network (hard update).
        An EMA variant via update_ema() from jku.wad/agents/utils.py is a
        valid alternative — swap the implementation here without changing callers.
        """
        raise NotImplementedError

    def train(self, env, logger: Logger) -> None:
        """
        Full training loop. Requires setup() to have been called first.

        Runs the episode/step loop until cfg.total_steps is reached.
        Saves checkpoints every cfg.checkpoint_interval steps and exports
        the final ONNX model to cfg.out_dir / cfg.run_name / submission.onnx.
        """
        raise NotImplementedError
