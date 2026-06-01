import copy
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..buffers.replay import ReplayBuffer
from ..config import DQNConfig
from ..models.dqn import DQNModel
from ..utils.logger import Logger
from .base import BaseAgent


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with epsilon-greedy exploration and hard target network sync.

    Training flow:
        agent.train(env, logger)
            └── step loop:
                    action = epsilon_greedy(obs)
                    env.step(action) → buffer.add(...)
                    if len(buffer) >= warmup_steps:
                        update()
                        if grad_steps % target_update_freq == 0:
                            _sync_target()
                    if done:
                        decay_epsilon() + log episode
    """

    def __init__(self, model: DQNModel, cfg: DQNConfig, device: torch.device):
        super().__init__(model, cfg, device)
        self.cfg: DQNConfig = cfg
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        self.target_model = copy.deepcopy(model).to(device)
        for p in self.target_model.parameters():
            p.requires_grad_(False)
        self._buffer: ReplayBuffer | None = None
        self._epsilon: float = cfg.epsilon_start
        self._grad_steps: int = 0

    def setup(self, obs_shape: tuple) -> None:
        """Allocate the ReplayBuffer. Called automatically by train()."""
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
        """Greedy evaluation action — argmax Q-value. obs: (C, H, W) CPU tensor."""
        obs = obs.unsqueeze(0).to(self.device, dtype=torch.float32)
        return self.model(obs).argmax(-1).cpu().item()

    def update(self) -> dict[str, float]:
        """
        One gradient step on a random minibatch from the replay buffer.

        Computes the Bellman target using the frozen target network:
            Q_target = r + γ * max_a Q_target(s', a) * (1 - done)

        Uses Huber loss (smooth_l1) for robustness to reward-scale outliers.
        Increments self._grad_steps; caller handles target sync frequency.

        Returns: loss, q_mean, epsilon, buffer_size.
        """
        if self._buffer is None:
            raise RuntimeError("Call setup() before update().")

        batch = self._buffer.sample(self.cfg.batch_size)

        self.model.train()

        # Q(s, a) for the actions actually taken
        q_current = (
            self.model(batch["obs"])
            .gather(1, batch["actions"].unsqueeze(1))
            .squeeze(1)
        )

        # Bellman target — no gradient through target network
        with torch.no_grad():
            q_next = self.target_model(batch["next_obs"]).max(dim=1)[0]
            q_target = batch["rewards"] + self.cfg.gamma * q_next * (1.0 - batch["dones"])

        loss = F.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._grad_steps += 1

        return {
            "loss": loss.item(),
            "q_mean": q_current.mean().item(),
            "epsilon": self._epsilon,
            "buffer_size": float(len(self._buffer)),
        }

    # ------------------------------------------------------------------
    # Exploration helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def epsilon_greedy(self, obs: torch.Tensor) -> int:
        """ε-greedy action for training. obs: (C, H, W) CPU tensor."""
        if random.random() < self._epsilon:
            return random.randrange(self.model.q_head.out_features)
        return self.select_action(obs)

    def decay_epsilon(self) -> None:
        """Multiplicative decay, clipped at cfg.epsilon_end. Call once per episode."""
        self._epsilon = max(self.cfg.epsilon_end, self._epsilon * self.cfg.epsilon_decay)

    def _sync_target(self) -> None:
        """Hard-copy online network weights into the target network."""
        self.target_model.load_state_dict(self.model.state_dict())

    # ------------------------------------------------------------------
    # Checkpoint override — persists target network and exploration state
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self._epsilon,
            "grad_steps": self._grad_steps,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.step = ckpt["step"]
        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt["target_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._epsilon = ckpt.get("epsilon", self._epsilon)
        self._grad_steps = ckpt.get("grad_steps", 0)
        print(f"Checkpoint loaded ← {path}  (step {self.step})")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, env, logger: Logger) -> None:
        """
        Full training loop. Logs at each episode end, checkpoints every
        cfg.checkpoint_interval steps, exports ONNX on completion.
        """
        # Use the actual post-transform shape, not observation_space which
        # reports raw resolution before the env's resize transform.
        obs = env.reset()[0]
        obs_shape = obs.shape
        if self._buffer is None:
            self.setup(obs_shape)
        ep_reward = 0.0
        ep_len = 0
        last_checkpoint = 0
        last_metrics: dict[str, float] = {}
        t_start = time.perf_counter()

        while self.step < self.cfg.total_steps:
            action = self.epsilon_greedy(obs)
            obs_list, rwds, done, _ = env.step(action)
            next_obs = obs_list[0]
            reward = rwds[0]

            self._buffer.add(obs, action, reward, next_obs, done)
            ep_reward += reward
            ep_len += 1
            self.step += 1

            obs = env.reset()[0] if done else next_obs

            # Gradient update (after warmup)
            if len(self._buffer) >= self.cfg.warmup_steps:
                last_metrics = self.update()
                if self._grad_steps % self.cfg.target_update_freq == 0:
                    self._sync_target()

            # Log at episode boundaries
            if done:
                fps = int(self.step / (time.perf_counter() - t_start))
                logger.log(
                    self.step,
                    fps=fps,
                    ep_reward=ep_reward,
                    ep_len=ep_len,
                    **last_metrics,
                )
                ep_reward, ep_len = 0.0, 0
                self.decay_epsilon()

            # Checkpoint
            if self.step - last_checkpoint >= self.cfg.checkpoint_interval:
                ckpt = Path(self.cfg.out_dir) / self.cfg.run_name / f"ckpt_{self.step:09d}.pt"
                self.save(ckpt)
                last_checkpoint = self.step

        out_dir = Path(self.cfg.out_dir) / self.cfg.run_name
        self.export_onnx(out_dir / "submission.onnx", obs_shape)
        env.close()
