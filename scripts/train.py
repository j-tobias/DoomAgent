"""
PPO training entry point.

Usage:
    uv run scripts/train.py --run-name ppo_baseline
    uv run scripts/train.py --run-name ppo_exp --total-steps 2000000 --n-steps 1024
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from doomagent.agents.ppo import PPOAgent
from doomagent.config import EnvConfig, PPOConfig
from doomagent.env import make_env
from doomagent.models.encoder import NatureCNN
from doomagent.models.ppo import PPOActorCritic
from doomagent.reward import CustomReward
from doomagent.utils.logger import Logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="ppo_run")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env_cfg = EnvConfig(seed=args.seed)
    cfg = PPOConfig(
        env=env_cfg,
        run_name=args.run_name,
        total_steps=args.total_steps,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        lr=args.lr,
    )

    env = make_env(env_cfg, reward_fn=CustomReward(num_players=1))
    n_actions = env.action_space.n
    # observation_space spatial dims are pre-transform; channels are correct
    in_channels = env.observation_space.shape[0]

    encoder = NatureCNN(in_channels=in_channels)
    model = PPOActorCritic(encoder, n_actions=n_actions, env_cfg=env_cfg)

    agent = PPOAgent(model, cfg, device)

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
