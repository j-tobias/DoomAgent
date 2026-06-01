"""
DQN training entry point.

Usage:
    uv run scripts/train_dqn.py --run-name dqn_baseline
    uv run scripts/train_dqn.py --run-name dqn_exp --total-steps 500000 --buffer-size 50000
"""
import argparse
from pathlib import Path

import torch

from doomagent.agents.dqn import DQNAgent
from doomagent.config import DQNConfig, EnvConfig
from doomagent.env import make_env
from doomagent.models.dqn import DQNModel
from doomagent.models.encoder import NatureCNN
from doomagent.reward import CustomReward
from doomagent.utils.logger import Logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="dqn_run")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--buffer-size", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env_cfg = EnvConfig(seed=args.seed)
    cfg = DQNConfig(
        env=env_cfg,
        run_name=args.run_name,
        total_steps=args.total_steps,
        buffer_size=args.buffer_size,
        lr=args.lr,
    )

    env = make_env(env_cfg, reward_fn=CustomReward(num_players=1))
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    encoder = NatureCNN(in_channels=obs_shape[0])
    model = DQNModel(encoder, n_actions=n_actions, env_cfg=env_cfg)
    agent = DQNAgent(model, cfg, device)

    log_dir = Path(cfg.out_dir) / cfg.run_name
    with Logger(
        log_dir,
        project=None if args.no_wandb else "doomagent",
        run_name=cfg.run_name,
        config=cfg,
    ) as logger:
        agent.train(env, logger)


if __name__ == "__main__":
    main()
