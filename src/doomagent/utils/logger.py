import csv
import dataclasses
import time
from pathlib import Path
from typing import Any, Optional


class Logger:
    """
    Logs metrics to console, a CSV file, and optionally wandb.

    The CSV schema is dynamic — new keys can appear at any log() call
    (e.g. ep_reward_mean only appears when an episode completes). When that
    happens the file is rewritten with the updated header and all prior rows
    backfilled with empty strings. For typical training runs (~2k log events)
    the rewrite cost is negligible.

    Usage without wandb:
        with Logger("runs/my_run") as logger:
            logger.log(step, loss=0.5, reward=12.3)

    Usage with wandb:
        with Logger("runs/my_run", project="doomagent", config=cfg) as logger:
            logger.log(step, loss=0.5, reward=12.3)
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
        self._t0 = time.time()

        self._rows: list[dict] = []
        self._fieldnames: list[str] = []
        self._file = None
        self._writer = None

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

        # Detect new keys and rewrite CSV if the schema changed
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            self._rewrite_csv()

        self._rows.append(row)
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

    def _rewrite_csv(self) -> None:
        if self._file:
            self._file.close()
        self._file = open(self._csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._fieldnames, restval=""
        )
        self._writer.writeheader()
        for row in self._rows:
            self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
