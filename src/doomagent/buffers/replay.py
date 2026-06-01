import torch


class ReplayBuffer:
    """
    Ring-buffer experience replay for DQN. Uniform random sampling.

    All tensors are pre-allocated on `device` at construction. The write
    pointer wraps modularly so old transitions are silently overwritten once
    capacity is reached.
    """

    def __init__(self, capacity: int, obs_shape: tuple, device: torch.device):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = device
        self._ptr = 0
        self._size = 0
        self._alloc()

    # ------------------------------------------------------------------
    # Storage — real implementations
    # ------------------------------------------------------------------

    def _alloc(self) -> None:
        n, s, d = self.capacity, self.obs_shape, self.device
        self._obs = torch.zeros(n, *s, device=d, dtype=torch.float32)
        self._next_obs = torch.zeros(n, *s, device=d, dtype=torch.float32)
        self._actions = torch.zeros(n, device=d, dtype=torch.int64)
        self._rewards = torch.zeros(n, device=d, dtype=torch.float32)
        self._dones = torch.zeros(n, device=d, dtype=torch.float32)

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
    ) -> None:
        """
        Write one transition at the current pointer position, then advance it.
        obs / next_obs are copied to self.device if needed.
        """
        i = self._ptr
        self._obs[i] = obs.to(self.device, dtype=torch.float32)
        self._next_obs[i] = next_obs.to(self.device, dtype=torch.float32)
        self._actions[i] = int(action)
        self._rewards[i] = float(reward)
        self._dones[i] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Draw batch_size transitions uniformly at random from the filled portion.

        Returns a dict with keys:
            obs        (B, *obs_shape) float32
            actions    (B,)            int64
            rewards    (B,)            float32
            next_obs   (B, *obs_shape) float32
            dones      (B,)            float32  — 1.0 if terminal
        """
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return {
            "obs": self._obs[idx],
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "next_obs": self._next_obs[idx],
            "dones": self._dones[idx],
        }

    def __len__(self) -> int:
        return self._size
