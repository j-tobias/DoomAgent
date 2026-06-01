from pathlib import Path

import doom_arena
from doom_arena.doom_env import VizdoomMPEnv
from doom_arena.player import ObsBuffer

from .config import EnvConfig

# Resolve once at import time — works regardless of cwd
_JKU_CFG = str(Path(doom_arena.__file__).parent / "scenarios" / "jku.cfg")


def make_env(cfg: EnvConfig) -> VizdoomMPEnv:
    extra_state = None
    if cfg.extra_state:
        extra_state = [ObsBuffer(s) for s in cfg.extra_state]

    return VizdoomMPEnv(
        config_path=_JKU_CFG,
        num_players=1,
        num_bots=cfg.num_bots,
        bot_skill=cfg.bot_skill,
        doom_map=cfg.doom_map,
        episode_timeout=cfg.episode_timeout,
        screen_format=cfg.screen_format,
        n_stack_frames=cfg.n_stack_frames,
        extra_state=extra_state,
        hud=cfg.hud,
        crosshair=cfg.crosshair,
        seed=cfg.seed,
    )
