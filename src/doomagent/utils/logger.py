import csv
import dataclasses
import time
from pathlib import Path
from typing import Any, Optional


class Logger:
    """
    Logs metrics to console, a CSV file, and optionally wandb.

    Usage without wandb:
        with Logger("runs/my_run") as logger:
            logger.log(step, loss=0.5, reward=12.3)

    Usage with wandb:
        with Logger("runs/my_run", project="doomagent", config=cfg) as logger:
            logger.log(step, loss=0.5, reward=12.3)

    `config` accepts a dataclass instance or a plain dict.
    wandb is only initialised if `project` is provided.
    """

    def __init__(
        self,
        log_dir: str | Path,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._dir / "metrics.csv"
        self._file = None
        self._writer = None
        self._t0 = time.time()
        self._wandb = None

        if project is not None:
            import wandb
            cfg_dict = (
                dataclasses.asdict(config)
                if dataclasses.is_dataclass(config)
                else (config or {})
            )
            self._wandb = wandb.init(
                project=project,
                name=run_name or self._dir.name,
                config=cfg_dict,
                dir=str(self._dir),
            )

    def log(self, step: int, **metrics) -> None:
        row = {"step": step, "elapsed_s": int(time.time() - self._t0), **metrics}

        # CSV
        if self._writer is None:
            self._file = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

        # wandb
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

        # console
        parts = "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()
        )
        print(f"[{self._dir.name}] {parts}")

    def close(self) -> None:
        if self._file:
            self._file.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
