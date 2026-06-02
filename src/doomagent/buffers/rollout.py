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
        dones      (n_steps,)  — 0.0 or 1.0, done flag returned by env.step()
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
    # Storage management
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
    # GAE computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_gae(self, last_value: torch.Tensor, last_done: bool) -> None:
        """
        Compute GAE(λ) advantages and discounted returns in-place.

        dones[t] = 1 means the episode ended after step t, so the next
        observation belongs to a new episode and should not be bootstrapped.

        Args:
            last_value: V(s_{T+1}) bootstrap estimate — shape (1,) or scalar.
            last_done:  True if the episode ended on the final collected step.
        """
        last_gae = torch.zeros(1, device=self.device)
        last_value = last_value.squeeze().to(self.device)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_nonterminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_nonterminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_nonterminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values
        self._gae_computed = True

    # ------------------------------------------------------------------
    # Minibatch iteration
    # ------------------------------------------------------------------

    def iter_minibatches(
        self, n_minibatches: int, compute_device: torch.device | None = None
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """
        Yield n_minibatches random-permuted minibatches.

        Advantages are normalised (mean=0, std=1) across the full buffer
        before splitting. Raises RuntimeError if compute_gae() has not
        been called since the last add() or reset().

        Args:
            n_minibatches:  number of minibatches to split the buffer into.
            compute_device: device to move each batch to before yielding.
                            Defaults to self.device (storage device).
                            Pass the model's device (e.g. cuda:0) when the
                            buffer is stored on CPU to avoid OOM on large buffers.

        Each yielded dict has keys:
            obs, actions, log_probs_old, values_old, advantages, returns
        """
        if not self._gae_computed:
            raise RuntimeError("Call compute_gae() before iter_minibatches().")

        target = compute_device or self.device
        batch_size = self.n_steps // n_minibatches

        # Normalise advantages globally before splitting into minibatches
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = torch.randperm(self.n_steps, device=self.device)

        for start in range(0, self.n_steps, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs":          self.obs[idx].to(target),
                "actions":      self.actions[idx].to(target),
                "log_probs_old":self.log_probs[idx].to(target),
                "values_old":   self.values[idx].to(target),
                "advantages":   adv[idx].to(target),
                "returns":      self.returns[idx].to(target),
            }
