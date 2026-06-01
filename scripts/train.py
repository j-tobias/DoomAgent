"""
Training entry point. Copy / adapt for each experiment.

Usage:
    uv run scripts/train.py --run-name ppo_baseline --total-steps 1000000
"""
import argparse
from pathlib import Path

import torch

from doomagent.config import EnvConfig, TrainConfig
from doomagent.env import make_env
from doomagent.utils.logger import Logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="run")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_cfg = EnvConfig(seed=args.seed)
    cfg = TrainConfig(
        env=env_cfg,
        total_steps=args.total_steps,
        run_name=args.run_name,
    )

    log_dir = Path(cfg.out_dir) / cfg.run_name

    with Logger(log_dir) as logger:
        env = make_env(cfg.env)
        obs_shape = env.observation_space.shape  # (C, H, W)

        # --- plug in your agent here ---
        # from doomagent.agents.ppo import PPOAgent
        # from doomagent.models.cnn import CNNModel
        # model = CNNModel(env_cfg, n_actions=env.action_space.n)
        # agent = PPOAgent(model, cfg, device)
        # agent.train(env, logger)
        # agent.export_onnx(log_dir / "submission.onnx", obs_shape)

        env.close()


if __name__ == "__main__":
    main()
