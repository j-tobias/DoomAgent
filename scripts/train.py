#!/usr/bin/env python -u
"""
PPO training entry point.  (-u flag disables stdout buffering for live logs)

Usage:
    uv run scripts/train.py --run-name ppo_baseline
    uv run scripts/train.py --run-name ppo_labels --extra-state labels
    uv run scripts/train.py --run-name ppo_entropy --ent-coef 0.05
    uv run scripts/train.py --run-name ppo_reward --death-penalty
    uv run scripts/train.py --run-name ppo_long --total-steps 2000000
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from doomagent.agents.ppo import PPOAgent
from doomagent.config import EnvConfig, PPOConfig
from doomagent.env import make_env
from doomagent.models.encoder import IMPALAEncoder, NatureCNN
from doomagent.models.ppo import PPOActorCritic
from doomagent.reward import AliveReward, CustomReward, DeathPenaltyReward
from doomagent.utils.logger import Logger

_ENCODERS = {"nature": NatureCNN, "impala": IMPALAEncoder}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="ppo_run")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--n-minibatches", type=int, default=None,
                   help="Minibatches per rollout. Defaults to n_steps//512 to keep batch_size~512")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--extra-state", nargs="+", default=None,
                   help="Extra observation buffers, e.g. --extra-state labels depth")
    p.add_argument("--n-stack-frames", type=int, default=1,
                   help="Number of frames to stack in the observation (default: 1)")
    p.add_argument("--death-penalty", action="store_true",
                   help="Use DeathPenaltyReward instead of CustomReward")
    p.add_argument("--death-penalty-value", type=float, default=10.0,
                   help="Penalty magnitude on death (default: 10.0)")
    p.add_argument("--alive-reward", action="store_true",
                   help="Use AliveReward: base reward + per-tick bonus for being alive")
    p.add_argument("--encoder", choices=["nature", "impala"], default="impala",
                   help="Encoder architecture (default: impala)")
    p.add_argument("--no-reward-norm", action="store_true",
                   help="Disable reward normalisation")
    p.add_argument("--no-ent-anneal", action="store_true",
                   help="Disable entropy coefficient annealing")
    p.add_argument("--ent-coef-final", type=float, default=0.001,
                   help="Final entropy coefficient when annealing (default: 0.001)")
    p.add_argument("--target-kl", type=float, default=0.01,
                   help="KL early-stopping threshold (default: 0.01, set 0 to disable)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-random-seeds", action="store_true",
                   help="Fix the VizDoom seed across all episodes (disables spawn randomisation)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--pretrain-checkpoint", type=str, default=None,
                   help="Checkpoint to warm-start model weights from (optimizer and step reset)")
    p.add_argument("--partial-load", action="store_true",
                   help="Skip layers whose shapes don't match (for architecture changes e.g. stack4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env_cfg = EnvConfig(seed=args.seed, extra_state=args.extra_state,
                        n_stack_frames=args.n_stack_frames)
    n_minibatches = args.n_minibatches or max(1, args.n_steps // 512)
    cfg = PPOConfig(
        env=env_cfg,
        run_name=args.run_name,
        total_steps=args.total_steps,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        n_minibatches=n_minibatches,
        lr=args.lr,
        ent_coef=args.ent_coef,
        normalize_rewards=not args.no_reward_norm,
        anneal_ent_coef=not args.no_ent_anneal,
        ent_coef_final=args.ent_coef_final,
        target_kl=args.target_kl if args.target_kl > 0 else None,
        random_seeds=not args.no_random_seeds,
    )

    if args.death_penalty:
        reward_fn = DeathPenaltyReward(num_players=1, death_penalty=args.death_penalty_value)
    elif args.alive_reward:
        reward_fn = AliveReward(num_players=1)
    else:
        reward_fn = CustomReward(num_players=1)
    env = make_env(env_cfg, reward_fn=reward_fn)
    n_actions = env.action_space.n
    # observation_space.shape reports raw VizDoom resolution (never updated for stacking)
    in_channels = env.observation_space.shape[0] * env_cfg.n_stack_frames

    encoder = _ENCODERS[args.encoder](in_channels=in_channels)
    model = PPOActorCritic(encoder, n_actions=n_actions, env_cfg=env_cfg)
    agent = PPOAgent(model, cfg, device)

    if args.pretrain_checkpoint:
        agent.load_weights(args.pretrain_checkpoint, partial=args.partial_load)
        print(f"Warm-started from {args.pretrain_checkpoint}")

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
