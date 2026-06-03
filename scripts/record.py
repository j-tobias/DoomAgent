#!/usr/bin/env python -u
"""
Record a video of the agent playing and optionally evaluate over N episodes.

Supports two loading modes:
    --submission model.onnx   Load via onnx2pytorch (same as grading server)
    --checkpoint  ckpt.pt     Load PyTorch weights directly (no onnx2pytorch)

Usage:
    uv run scripts/record.py --checkpoint runs/impala_6M/ckpt_006000000.pt --record-best
    uv run scripts/record.py --submission runs/impala_6M/submission.onnx --episodes 5
"""
import argparse
import os
import sys

import imageio
import numpy as np
import torch
from torch.distributions.categorical import Categorical

# Import shared loading helpers from evaluate.py (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import load_from_checkpoint, load_from_onnx, _Agent

from doomagent.config import EnvConfig
from doomagent.env import make_env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def run_episode(agent: _Agent, env_cfg: EnvConfig, record: bool = False):
    env = make_env(env_cfg)
    if record:
        env.enable_replay()

    obs = env.reset()[0]
    score, done = 0.0, False
    while not done:
        action = agent.select_action(obs)
        obs_list, rwds, done, _ = env.step(action)
        obs = obs_list[0]
        score += rwds[0]

    replays = env.get_player_replays() if record else {}
    env.close()
    return score, replays


def frames_to_video(replays: dict, path: str, fps: int = 35) -> None:
    """Convert raw replay frames (C, H, W) BGR uint8 → RGB MP4."""
    player_replay = next(iter(replays.values()))
    raw_frames = player_replay.get("frames", [])
    if not raw_frames:
        print("No frames captured.")
        return

    video_frames = []
    for f in raw_frames:
        f = np.asarray(f)
        if f.ndim == 3 and f.shape[0] in (1, 3, 4):
            f = f.transpose(1, 2, 0)           # (H, W, C)
            if f.shape[2] == 3:
                f = f[:, :, ::-1].copy()        # BGR → RGB
            elif f.shape[2] == 1:
                f = np.repeat(f, 3, axis=2)
        video_frames.append(f.astype(np.uint8))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimwrite(path, video_frames, fps=fps, quality=8, macro_block_size=1)
    duration = len(video_frames) / fps
    print(f"Video saved → {path}  ({len(video_frames)} frames, {duration:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submission", help="ONNX model path")
    group.add_argument("--checkpoint", help="PyTorch checkpoint path (.pt)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--record-best", action="store_true",
                        help="Save MP4 of the best scoring episode")
    parser.add_argument("--out", type=str, default=None,
                        help="Video output path (default: next to model file)")
    args = parser.parse_args()

    if args.submission:
        model, config = load_from_onnx(args.submission, DEVICE)
        out_dir = os.path.dirname(args.submission)
        print(f"Loaded ONNX from {args.submission}")
    else:
        model, config = load_from_checkpoint(args.checkpoint, DEVICE)
        out_dir = os.path.dirname(args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")

    agent = _Agent(model, config, DEVICE)
    env_cfg = EnvConfig(
        screen_format=config.get("screen_format", 0),
        n_stack_frames=config.get("n_stack_frames", 1),
        extra_state=config.get("extra_state"),
        hud=config.get("hud", "full"),
        crosshair=config.get("crosshair", True),
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    scores = []
    best_score, best_replays = -float("inf"), {}

    for i in range(args.episodes):
        score, replays = run_episode(agent, env_cfg, record=args.record_best)
        scores.append(score)
        print(f"Episode {i + 1:2d}: {score:+.1f}")
        if score > best_score:
            best_score, best_replays = score, replays
        env_cfg.seed = int(rng.integers(int(1e7)))

    print(f"\nMean over {args.episodes} episodes: {np.mean(scores):+.2f}")
    print(f"Best episode:                      {max(scores):+.2f}")

    if args.record_best and best_replays:
        video_path = args.out or os.path.join(out_dir, "replay_best.mp4")
        frames_to_video(best_replays, video_path)


if __name__ == "__main__":
    main()
