from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvConfig:
    screen_format: int = 0                   # vzd.ScreenFormat value — 0=RGB, 1=grayscale
    n_stack_frames: int = 1
    extra_state: Optional[list[str]] = None  # e.g. ["depth"] or ["labels", "automap"]
    hud: str = "full"                        # "full" | "minimal" | "none"
    crosshair: bool = True
    doom_map: str = "ROOM"                   # "ROOM" | "TRNM" | "TRNMBIG"
    num_bots: int = 4
    bot_skill: int = 0                       # 0=easy, 1=medium, 2+=hard
    episode_timeout: int = 2000
    seed: int = 1337


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    algo_type: str = "POLICY"       # "POLICY" → Categorical sample; else argmax
    total_steps: int = 1_000_000
    log_interval: int = 1_000
    checkpoint_interval: int = 50_000
    run_name: str = "run"
    out_dir: str = "runs"
