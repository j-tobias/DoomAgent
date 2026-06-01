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


@dataclass
class PPOConfig(TrainConfig):
    algo_type: str = "POLICY"
    n_steps: int = 512              # env steps per rollout
    n_epochs: int = 4               # gradient epochs per rollout
    n_minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    anneal_lr: bool = True          # linear lr decay to 0 over total_steps


@dataclass
class DQNConfig(TrainConfig):
    algo_type: str = "QVALUE"
    buffer_size: int = 10_000
    batch_size: int = 32
    lr: float = 1e-4
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.99     # multiplicative per episode
    target_update_freq: int = 100   # gradient steps between target syncs
    warmup_steps: int = 1_000       # replay buffer steps before first update
