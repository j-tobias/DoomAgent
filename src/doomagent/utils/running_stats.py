import math


class RunningMeanStd:
    """
    Tracks the running mean and variance of a scalar stream using
    Welford's online algorithm (numerically stable, single-pass).

    Used for reward normalisation: dividing rewards by the running std
    keeps the value function's learning target on a consistent scale
    regardless of how large returns grow during training.

    Only std-normalisation is applied (no mean subtraction) so the
    sign and relative ordering of rewards is preserved.
    """

    def __init__(self, epsilon: float = 1e-4):
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = epsilon  # avoids div-by-zero on first update

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var = self.var + (delta * delta2 - self.var) / self.count

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 1e-8))

    def normalize(self, x: float, clip: float = 10.0) -> float:
        """Divide by running std and clip to [-clip, clip]."""
        return max(-clip, min(clip, x / self.std))

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]
