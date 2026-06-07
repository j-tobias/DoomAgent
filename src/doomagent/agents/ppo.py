import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ..buffers.rollout import RolloutBuffer
from ..config import PPOConfig
from ..env import reseed_env
from ..models.ppo import PPOActorCritic
from ..utils.logger import Logger
from ..utils.running_stats import RunningMeanStd
from .base import BaseAgent


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimisation agent.

    Training flow:
        agent.train(env, logger)
            └── while step < total_steps:
                    collect(env, obs)   # fill buffer, compute GAE
                    update(ent_coef)    # n_epochs (with KL early stop) × n_minibatches
                    log / checkpoint
    """

    def __init__(self, model: PPOActorCritic, cfg: PPOConfig, device: torch.device):
        super().__init__(model, cfg, device)
        self.cfg: PPOConfig = cfg
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        self._buffer: RolloutBuffer | None = None
        self._reward_rms = RunningMeanStd()  # for reward normalisation
        self._ema_reward: float | None = None
        self._best_ema_reward: float = float("-inf")

    def setup(self, obs_shape: tuple) -> None:
        """Allocate the RolloutBuffer on CPU (minibatches move to GPU during update)."""
        self._buffer = RolloutBuffer(
            n_steps=self.cfg.n_steps,
            obs_shape=obs_shape,
            device=torch.device("cpu"),
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

    # ------------------------------------------------------------------
    # Checkpoint — override base to persist reward RMS and entropy state
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "reward_rms": self._reward_rms.state_dict(),
        }, path)
        print(f"Checkpoint saved → {path}")

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.step = ckpt["step"]
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "reward_rms" in ckpt:
            self._reward_rms.load_state_dict(ckpt["reward_rms"])
        print(f"Checkpoint loaded ← {path}  (step {self.step})")

    def load_weights(self, path: str | Path, partial: bool = False) -> None:
        """Warm-start from a pretrained checkpoint — loads model weights and
        reward normalisation stats, but resets step counter and optimizer so
        LR annealing restarts cleanly for the fine-tuning budget."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if partial:
            current = self.model.state_dict()
            compatible = {k: v for k, v in ckpt["model"].items()
                          if k in current and v.shape == current[k].shape}
            current.update(compatible)
            self.model.load_state_dict(current)
            print(f"Partial load: {len(compatible)}/{len(ckpt['model'])} layers transferred")
        else:
            self.model.load_state_dict(ckpt["model"])
        if "reward_rms" in ckpt:
            self._reward_rms.load_state_dict(ckpt["reward_rms"])
        # step and optimizer intentionally left at initial values

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> int:
        """Deterministic evaluation action — argmax of policy logits."""
        obs = obs.unsqueeze(0).to(self.device, dtype=torch.float32)
        logits, _ = self.model(obs)
        return logits.argmax(-1).cpu().item()

    def update(self, ent_coef: float | None = None) -> dict[str, float]:
        """
        PPO gradient update with KL early stopping.

        Runs up to cfg.n_epochs epochs. If cfg.target_kl is set, stops
        early once the mean KL across a full epoch exceeds the threshold —
        preventing the policy from moving too far in a single update.

        Args:
            ent_coef: entropy coefficient for this update (allows annealing
                      from train()). Falls back to cfg.ent_coef if None.
        """
        if ent_coef is None:
            ent_coef = self.cfg.ent_coef

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
        epochs_done = 0

        for _ in range(self.cfg.n_epochs):
            epoch_kl = 0.0
            epoch_batches = 0

            for batch in self._buffer.iter_minibatches(self.cfg.n_minibatches, self.device):
                log_prob, entropy, value = self.model.evaluate_actions(
                    batch["obs"], batch["actions"]
                )
                value = value.squeeze()

                ratio = torch.exp(log_prob - batch["log_probs_old"])
                adv = batch["advantages"]
                policy_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps),
                ).mean()

                if self.cfg.clip_vf:
                    v_clipped = batch["values_old"] + torch.clamp(
                        value - batch["values_old"],
                        -self.cfg.clip_eps,
                        self.cfg.clip_eps,
                    )
                    value_loss = 0.5 * torch.max(
                        (value - batch["returns"]).pow(2),
                        (v_clipped - batch["returns"]).pow(2),
                    ).mean()
                else:
                    value_loss = 0.5 * (value - batch["returns"]).pow(2).mean()

                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    - ent_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.cfg.clip_eps).float().mean().item()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy_mean.item()
                totals["total_loss"] += loss.item()
                totals["approx_kl"] += approx_kl
                totals["clip_frac"] += clip_frac
                n_updates += 1
                epoch_kl += approx_kl
                epoch_batches += 1

            epochs_done += 1

            # KL early stopping — check after each full epoch
            if self.cfg.target_kl is not None and epoch_kl / epoch_batches > self.cfg.target_kl:
                break

        metrics = {k: v / n_updates for k, v in totals.items()}
        metrics["epochs_done"] = float(epochs_done)
        metrics["ent_coef"] = ent_coef
        return metrics

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def collect(self, env, obs: torch.Tensor) -> tuple[torch.Tensor, bool, dict[str, float]]:
        """
        Collect cfg.n_steps transitions. Rewards are normalised by running
        std before being stored in the buffer; ep_reward_mean is logged in
        raw units so the numbers remain interpretable.
        """
        if self._buffer is None:
            raise RuntimeError("Call setup(obs_shape) before collect().")

        self._buffer.reset()
        self.model.eval()

        ep_reward = 0.0     # raw reward for logging
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

                # Reward normalisation: update running stats, store normalised value
                if self.cfg.normalize_rewards:
                    self._reward_rms.update(reward)
                    stored_reward = self._reward_rms.normalize(reward)
                else:
                    stored_reward = reward

                self._buffer.add(obs, action[0], log_prob[0], stored_reward, done, value[0])
                self.step += 1

                if done:
                    completed.append({"ep_reward": ep_reward, "ep_len": ep_len})
                    ep_reward, ep_len = 0.0, 0
                    if self.cfg.random_seeds:
                        reseed_env(env, int(torch.randint(int(1e7), (1,)).item()))
                    obs = env.reset()[0]
                else:
                    obs = obs_list[0]

            last_value = self.model(obs.unsqueeze(0).to(self.device, dtype=torch.float32))[1]

        self._buffer.compute_gae(last_value, done)

        rollout_info: dict[str, float] = {}
        if completed:
            rollout_info["ep_reward_mean"] = sum(e["ep_reward"] for e in completed) / len(completed)
            rollout_info["ep_len_mean"]    = sum(e["ep_len"]    for e in completed) / len(completed)
            rollout_info["n_episodes"]     = float(len(completed))
            if self.cfg.normalize_rewards:
                rollout_info["reward_rms_std"] = self._reward_rms.std

        return obs, done, rollout_info

    def train(self, env, logger: Logger) -> None:
        """
        Full training loop with LR annealing, entropy annealing, reward
        normalisation, and KL early stopping.
        """
        obs = env.reset()[0]
        obs_shape = obs.shape
        if self._buffer is None:
            self.setup(obs_shape)
        last_checkpoint = 0
        out_dir = Path(self.cfg.out_dir) / self.cfg.run_name

        while self.step < self.cfg.total_steps:
            progress = self.step / self.cfg.total_steps

            # LR annealing with floor
            if self.cfg.anneal_lr:
                lr_frac = max(self.cfg.anneal_lr_min_frac, 1.0 - progress)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr_frac * self.cfg.lr

            # Entropy coefficient annealing
            if self.cfg.anneal_ent_coef:
                ent_coef = self.cfg.ent_coef_final + (
                    self.cfg.ent_coef - self.cfg.ent_coef_final
                ) * max(0.0, 1.0 - progress)
            else:
                ent_coef = self.cfg.ent_coef

            t0 = time.perf_counter()
            obs, done, rollout_info = self.collect(env, obs)
            update_metrics = self.update(ent_coef=ent_coef)
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

            # Best-checkpoint tracking: export submission.onnx whenever EMA reward peaks
            ep_r = rollout_info.get("ep_reward_mean")
            if ep_r is not None:
                self._ema_reward = (ep_r if self._ema_reward is None
                                    else 0.05 * ep_r + 0.95 * self._ema_reward)
                if (self.step > self.cfg.total_steps * 0.10
                        and self._ema_reward > self._best_ema_reward):
                    self._best_ema_reward = self._ema_reward
                    self.export_onnx(out_dir / "submission.onnx", obs_shape)

        self.export_onnx(out_dir / "submission.onnx", obs_shape)
        env.close()
