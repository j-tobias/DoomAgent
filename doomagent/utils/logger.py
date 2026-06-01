import csv
import time
from pathlib import Path


class Logger:
    """Writes metrics to console and a CSV file under log_dir/metrics.csv."""

    def __init__(self, log_dir: str | Path):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._dir / "metrics.csv"
        self._file = None
        self._writer = None
        self._t0 = time.time()

    def log(self, step: int, **metrics) -> None:
        row = {"step": step, "elapsed_s": int(time.time() - self._t0), **metrics}
        if self._writer is None:
            self._file = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()
        parts = "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()
        )
        print(f"[{self._dir.name}] {parts}")

    def close(self) -> None:
        if self._file:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
