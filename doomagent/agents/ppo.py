import torch
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
        agent.setup(obs_shape)          # allocate RolloutBuffer
        agent.train(env, logger)
            └── while step < total_steps:
                    collect(env, obs)   # fill buffer, compute GAE
                    update()            # n_epochs × n_minibatches gradient steps
                    log / checkpoint

    collect() and update() are separated so they can be extended or replaced
    independently (e.g. vectorised collection, mixed-precision update).
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
        Allocate the RolloutBuffer. Call once after obs_shape is known from
        env.observation_space.shape. Separated from __init__ so the agent
        can be constructed (and checkpointed) before an env exists.
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
        obs: (C, H, W) CPU tensor.

        During training use model.act() instead to obtain log_probs.
        """
        obs = obs.unsqueeze(0).to(self.device, dtype=torch.float32)
        logits, _ = self.model(obs)
        return logits.argmax(-1).cpu().item()

    def update(self) -> dict[str, float]:
        """
        Run cfg.n_epochs × cfg.n_minibatches gradient steps on the filled buffer.

        Requires collect() to have been called first.

        Returns a dict with keys:
            policy_loss, value_loss, entropy_loss, total_loss,
            approx_kl, clip_frac
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def collect(
        self, env, obs: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """
        Step the environment for cfg.n_steps, store transitions in the buffer,
        then call buffer.compute_gae() to prepare advantages.

        Args:
            env: VizdoomMPEnv (single player).
            obs: Current observation (C, H, W) CPU tensor — the starting state.

        Returns:
            (next_obs, done) — the state and done flag after the last step,
            to be passed as obs into the next collect() call.

        Raises RuntimeError if setup() has not been called.
        """
        if self._buffer is None:
            raise RuntimeError("Call setup(obs_shape) before collect().")
        raise NotImplementedError

    def train(self, env, logger: Logger) -> None:
        """
        Full training loop. Requires setup() to have been called first.

        Alternates collect() / update() until cfg.total_steps is reached.
        Saves checkpoints every cfg.checkpoint_interval steps and exports
        the final ONNX model to cfg.out_dir / cfg.run_name / submission.onnx.
        """
        raise NotImplementedError
