"""
Local evaluation — mirrors the grading server logic exactly.

Usage:
    uv run scripts/evaluate.py --submission runs/my_run/submission.onnx --episodes 10
"""
import argparse
import json
import os

import numpy as np
import onnx
import torch
from onnx2pytorch import ConvertModel
from torch.distributions.categorical import Categorical

from doomagent.config import EnvConfig
from doomagent.env import make_env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


class _OnnxAgent:
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


def run_episode(agent: _OnnxAgent, env_cfg: EnvConfig) -> float:
    env = make_env(env_cfg)
    obs = env.reset()[0]
    score, done = 0.0, False
    while not done:
        action = agent.select_action(obs)
        obs_list, rwds, done, _ = env.step(action)
        obs = obs_list[0]
        score += rwds[0]
    env.close()
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    size_mb = os.path.getsize(args.submission) / 1024 ** 2
    if size_mb > 50:
        raise ValueError(f"Model is {size_mb:.1f} MB — exceeds the 50 MB limit.")

    onnx_model = onnx.load(args.submission)
    config = next(
        (json.loads(p.value) for p in onnx_model.metadata_props if p.key == "config"),
        {},
    )
    model = ConvertModel(onnx_model).eval().to(DEVICE, dtype=DTYPE)
    agent = _OnnxAgent(model, config, DEVICE)

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
    for i in range(args.episodes):
        score = run_episode(agent, env_cfg)
        print(f"Episode {i + 1:2d}: {score:+.1f}")
        scores.append(score)
        env_cfg.seed = int(rng.integers(int(1e7)))

    print(f"\nMean over {args.episodes} episodes: {np.mean(scores):+.2f}")


if __name__ == "__main__":
    main()
