from typing import Generator

import torch


class RolloutBuffer:
    """
    Fixed-size on-device buffer for PPO rollout collection.

    All tensors are pre-allocated on `device` at construction so there is no
    per-step CPU↔GPU transfer beyond the obs copy in add().

    Lifecycle per training iteration:
        1. reset()                                  — zero the write pointer
        2. add(...) × n_steps                       — fill the buffer
        3. compute_gae(last_value, last_done)        — compute advantages/returns
        4. iter_minibatches(n_minibatches)           — yield batches for update

    Fields stored (all float32 unless noted):
        obs        (n_steps, *obs_shape)
        actions    (n_steps,)  int64
        log_probs  (n_steps,)
        rewards    (n_steps,)
        dones      (n_steps,)  — 0.0 or 1.0
        values     (n_steps,)
        advantages (n_steps,)  — filled by compute_gae()
        returns    (n_steps,)  — filled by compute_gae()
    """

    def __init__(
        self,
        n_steps: int,
        obs_shape: tuple,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.n_steps = n_steps
        self.obs_shape = obs_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._ptr = 0
        self._gae_computed = False
        self._alloc()

    # ------------------------------------------------------------------
    # Storage management — real implementations
    # ------------------------------------------------------------------

    def _alloc(self) -> None:
        n, s, d = self.n_steps, self.obs_shape, self.device
        self.obs = torch.zeros(n, *s, device=d, dtype=torch.float32)
        self.actions = torch.zeros(n, device=d, dtype=torch.int64)
        self.log_probs = torch.zeros(n, device=d)
        self.rewards = torch.zeros(n, device=d)
        self.dones = torch.zeros(n, device=d)
        self.values = torch.zeros(n, device=d)
        self.advantages = torch.zeros(n, device=d)
        self.returns = torch.zeros(n, device=d)

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
    ) -> None:
        """
        Store one transition. obs is copied to self.device if needed.
        Raises RuntimeError if the buffer is already full.
        """
        if self._ptr >= self.n_steps:
            raise RuntimeError("RolloutBuffer is full — call reset() before adding.")
        i = self._ptr
        self.obs[i] = obs.to(self.device, dtype=torch.float32)
        self.actions[i] = action.to(self.device)
        self.log_probs[i] = log_prob.to(self.device)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.values[i] = value.squeeze().to(self.device)
        self._ptr += 1
        self._gae_computed = False

    @property
    def is_full(self) -> bool:
        return self._ptr == self.n_steps

    def reset(self) -> None:
        self._ptr = 0
        self._gae_computed = False

    # ------------------------------------------------------------------
    # GAE computation — STUB (training logic)
    # ------------------------------------------------------------------

    def compute_gae(self, last_value: torch.Tensor, last_done: bool) -> None:
        """
        Compute GAE(λ) advantages and discounted returns in-place.

        Must be called once after the buffer is full and before
        iter_minibatches(). Sets self.advantages and self.returns.

        Args:
            last_value: V(s_{T+1}) bootstrap estimate, shape (1,) or scalar.
                        Pass zeros if the episode ended (last_done=True).
            last_done:  True if the final step was a terminal episode step.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Minibatch iteration — STUB (depends on GAE output)
    # ------------------------------------------------------------------

    def iter_minibatches(
        self, n_minibatches: int
    ) -> "Generator[dict[str, torch.Tensor], None, None]":
        """
        Yield n_minibatches random-permuted minibatches as dicts:
            obs, actions, log_probs_old, advantages, returns

        Advantages are normalised (mean=0, std=1) across the full buffer
        before splitting. Raises RuntimeError if compute_gae() has not
        been called since the last add() or reset().

        batch_size = n_steps // n_minibatches (integer division).
        """
        if not self._gae_computed:
            raise RuntimeError(
                "Call compute_gae() before iter_minibatches()."
            )
        raise NotImplementedError
