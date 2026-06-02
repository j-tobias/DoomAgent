#!/usr/bin/env python -u
"""
Record a video of the agent playing one episode and evaluate over N episodes.

Usage:
    uv run scripts/record.py --submission runs/impala_6M/submission.onnx
    uv run scripts/record.py --submission runs/impala_6M/submission.onnx --episodes 5 --record-best
"""
import argparse
import json
import os

import imageio
import numpy as np
import onnx
import torch
from onnx2pytorch import ConvertModel
from torch.distributions.categorical import Categorical

from doomagent.config import EnvConfig
from doomagent.env import make_env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


class _Agent:
    def __init__(self, model, config: dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> int:
        obs = obs.unsqueeze(0).to(self.device, dtype=DTYPE)
        logits = self.model(obs)
        if isinstance(logits, tuple):
            logits, _ = logits
        if self.config.get("algo_type", "POLICY") == "POLICY":
            return Categorical(logits=logits).sample().cpu().item()
        return logits.argmax(-1).cpu().item()


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

    # (C, H, W) BGR → (H, W, C) RGB
    video_frames = []
    for f in raw_frames:
        f = np.asarray(f)
        if f.ndim == 3 and f.shape[0] in (1, 3, 4):
            f = f.transpose(1, 2, 0)          # (H, W, C)
            if f.shape[2] == 3:
                f = f[:, :, ::-1].copy()      # BGR → RGB
            elif f.shape[2] == 1:
                f = np.repeat(f, 3, axis=2)   # grayscale → RGB
        video_frames.append(f.astype(np.uint8))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimwrite(path, video_frames, fps=fps, quality=8, macro_block_size=1)
    duration = len(video_frames) / fps
    print(f"Video saved → {path}  ({len(video_frames)} frames, {duration:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--record-best", action="store_true",
                        help="Record a video of the best scoring episode")
    parser.add_argument("--out", type=str, default=None,
                        help="Video output path (default: next to submission)")
    args = parser.parse_args()

    size_mb = os.path.getsize(args.submission) / 1024 ** 2
    if size_mb > 50:
        raise ValueError(f"Model is {size_mb:.1f} MB — exceeds the 50 MB limit.")

    onnx_model = onnx.load(args.submission)
    config = next(
        (json.loads(p.value) for p in onnx_model.metadata_props if p.key == "config"), {}
    )
    model = ConvertModel(onnx_model).eval().to(DEVICE, dtype=DTYPE)
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
        # Record all episodes if --record-best, to capture the best
        score, replays = run_episode(agent, env_cfg, record=args.record_best)
        scores.append(score)
        print(f"Episode {i + 1:2d}: {score:+.1f}")
        if score > best_score:
            best_score, best_replays = score, replays
        env_cfg.seed = int(rng.integers(int(1e7)))

    print(f"\nMean over {args.episodes} episodes: {np.mean(scores):+.2f}")
    print(f"Best episode:                      {max(scores):+.2f}")

    if args.record_best and best_replays:
        video_path = args.out or os.path.join(
            os.path.dirname(args.submission), "replay_best.mp4"
        )
        frames_to_video(best_replays, video_path)


if __name__ == "__main__":
    main()
