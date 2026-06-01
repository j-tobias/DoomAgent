import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ..buffers.rollout import RolloutBuffer
from ..config import PPOConfig
from ..models.ppo import PPOActorCritic
from ..utils.logger import Logger
from .base import BaseAgent


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimisation agent.

    Training flow:
        agent.train(env, logger)
            └── while step < total_steps:
                    collect(env, obs)   # fill buffer, compute GAE
                    update()            # n_epochs × n_minibatches gradient steps
                    log / checkpoint
    """

    def __init__(
        self,
        model: PPOActorCritic,
        cfg: PPOConfig,
        device: torch.device,
    ):
        super().__init__(model, cfg, device)
        self.cfg: PPOConfig = cfg
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        self._buffer: RolloutBuffer | None = None

    def setup(self, obs_shape: tuple) -> None:
        """
        Allocate the RolloutBuffer. Called automatically by train() but can
        be called manually beforehand when obs_shape is already known.
        """
        self._buffer = RolloutBuffer(
            n_steps=self.cfg.n_steps,
            obs_shape=obs_shape,
            device=self.device,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> int:
        """
        Deterministic evaluation action — argmax of policy logits.
        obs: (C, H, W) CPU tensor. During training use model.act() instead.
        """
        obs = obs.unsqueeze(0).to(self.device, dtype=torch.float32)
        logits, _ = self.model(obs)
        return logits.argmax(-1).cpu().item()

    def update(self) -> dict[str, float]:
        """
        Run cfg.n_epochs × cfg.n_minibatches PPO gradient steps on the buffer.

        Returns averaged metrics over all gradient steps:
            policy_loss, value_loss, entropy, total_loss, approx_kl, clip_frac
        """
        self.model.train()

        totals: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }
        n_updates = 0

        for _ in range(self.cfg.n_epochs):
            for batch in self._buffer.iter_minibatches(self.cfg.n_minibatches):
                log_prob, entropy, value = self.model.evaluate_actions(
                    batch["obs"], batch["actions"]
                )
                value = value.squeeze()

                # PPO clipped surrogate objective
                ratio = torch.exp(log_prob - batch["log_probs_old"])
                adv = batch["advantages"]
                policy_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps),
                ).mean()

                # Value loss
                value_loss = 0.5 * (value - batch["returns"]).pow(2).mean()

                # Entropy bonus
                entropy_mean = entropy.mean()

                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    - self.cfg.ent_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clip_frac = ((ratio - 1).abs() > self.cfg.clip_eps).float().mean()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy_mean.item()
                totals["total_loss"] += loss.item()
                totals["approx_kl"] += approx_kl.item()
                totals["clip_frac"] += clip_frac.item()
                n_updates += 1

        return {k: v / n_updates for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def collect(
        self, env, obs: torch.Tensor
    ) -> tuple[torch.Tensor, bool, dict[str, float]]:
        """
        Step the environment for cfg.n_steps, store transitions in the buffer,
        then call buffer.compute_gae().

        Args:
            env: VizdoomMPEnv (single player).
            obs: Current observation (C, H, W) CPU tensor.

        Returns:
            obs:         Observation to start the next collect() call from.
            done:        Done flag of the final environment step.
            rollout_info: Dict with ep_reward_mean / ep_len_mean / n_episodes
                          for episodes that completed during this rollout.
        """
        if self._buffer is None:
            raise RuntimeError("Call setup(obs_shape) before collect().")

        self._buffer.reset()
        self.model.eval()

        ep_reward = 0.0
        ep_len = 0
        completed: list[dict] = []

        with torch.no_grad():
            for _ in range(self.cfg.n_steps):
                obs_t = obs.unsqueeze(0).to(self.device, dtype=torch.float32)
                action, log_prob, _, value = self.model.act(obs_t)

                obs_list, rwds, done, _ = env.step(action[0].cpu().item())
                reward = rwds[0]
                ep_reward += reward
                ep_len += 1

                self._buffer.add(obs, action[0], log_prob[0], reward, done, value[0])
                self.step += 1

                if done:
                    completed.append({"ep_reward": ep_reward, "ep_len": ep_len})
                    ep_reward, ep_len = 0.0, 0
                    obs = env.reset()[0]
                else:
                    obs = obs_list[0]

            # Bootstrap value for GAE
            last_value = self.model(obs.unsqueeze(0).to(self.device, dtype=torch.float32))[1]

        self._buffer.compute_gae(last_value, done)

        rollout_info: dict[str, float] = {}
        if completed:
            rollout_info["ep_reward_mean"] = sum(e["ep_reward"] for e in completed) / len(completed)
            rollout_info["ep_len_mean"] = sum(e["ep_len"] for e in completed) / len(completed)
            rollout_info["n_episodes"] = float(len(completed))

        return obs, done, rollout_info

    def train(self, env, logger: Logger) -> None:
        """
        Full training loop.

        Sets up the buffer from env.observation_space, then alternates
        collect() / update() until cfg.total_steps is reached.
        Logs every rollout, checkpoints every cfg.checkpoint_interval steps,
        and exports the final ONNX model on completion.
        """
        obs_shape = env.observation_space.shape
        if self._buffer is None:
            self.setup(obs_shape)

        obs = env.reset()[0]
        last_checkpoint = 0

        while self.step < self.cfg.total_steps:
            # Linear LR annealing
            if self.cfg.anneal_lr:
                frac = 1.0 - self.step / self.cfg.total_steps
                for pg in self.optimizer.param_groups:
                    pg["lr"] = frac * self.cfg.lr

            t0 = time.perf_counter()
            obs, done, rollout_info = self.collect(env, obs)
            update_metrics = self.update()
            fps = int(self.cfg.n_steps / (time.perf_counter() - t0))

            logger.log(
                self.step,
                fps=fps,
                lr=self.optimizer.param_groups[0]["lr"],
                **update_metrics,
                **rollout_info,
            )

            if self.step - last_checkpoint >= self.cfg.checkpoint_interval:
                ckpt = Path(self.cfg.out_dir) / self.cfg.run_name / f"ckpt_{self.step:09d}.pt"
                self.save(ckpt)
                last_checkpoint = self.step

        out_dir = Path(self.cfg.out_dir) / self.cfg.run_name
        self.export_onnx(out_dir / "submission.onnx", obs_shape)
        env.close()
