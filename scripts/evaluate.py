"""
Local evaluation — mirrors the grading server logic exactly.

Supports two loading modes:
    --submission model.onnx   Load via onnx2pytorch (same path as grading server)
    --checkpoint  ckpt.pt     Load PyTorch weights directly (faster, no onnx2pytorch)

Usage:
    uv run scripts/evaluate.py --submission runs/impala_6M/submission.onnx
    uv run scripts/evaluate.py --checkpoint runs/impala_6M/ckpt_006000000.pt
"""
import argparse
import json
import os

import numpy as np
import torch
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


def load_from_onnx(path: str, device: torch.device):
    import onnx as onnx_lib
    from onnx2pytorch import ConvertModel
    onnx_model = onnx_lib.load(path)
    config = next(
        (json.loads(p.value) for p in onnx_model.metadata_props if p.key == "config"), {}
    )
    model = ConvertModel(onnx_model).eval().to(device, dtype=DTYPE)
    return model, config


def load_from_checkpoint(path: str, device: torch.device):
    from doomagent.models.encoder import IMPALAEncoder, NatureCNN
    from doomagent.models.ppo import PPOActorCritic
    from doomagent.config import EnvConfig as EC
    ckpt = torch.load(path, map_location=device, weights_only=True)
    # Infer in_channels from first conv weight shape
    first_w = next(v for k, v in ckpt["model"].items() if "conv" in k and v.ndim == 4)
    in_channels = first_w.shape[1]
    # Try IMPALA first (smaller out_dim=256), fall back to NatureCNN
    env_cfg = EC()
    try:
        enc = IMPALAEncoder(in_channels=in_channels)
        model = PPOActorCritic(enc, n_actions=8, env_cfg=env_cfg)
        model.load_state_dict(ckpt["model"])
    except RuntimeError:
        enc = NatureCNN(in_channels=in_channels)
        model = PPOActorCritic(enc, n_actions=8, env_cfg=env_cfg)
        model.load_state_dict(ckpt["model"])
    model.eval().to(device, dtype=DTYPE)
    config = model.onnx_config
    return model, config


def run_episode(agent: _Agent, env_cfg: EnvConfig) -> float:
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submission", help="Path to ONNX submission file")
    group.add_argument("--checkpoint", help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.submission:
        size_mb = os.path.getsize(args.submission) / 1024 ** 2
        if size_mb > 50:
            raise ValueError(f"Model is {size_mb:.1f} MB — exceeds the 50 MB limit.")
        model, config = load_from_onnx(args.submission, DEVICE)
        print(f"Loaded ONNX from {args.submission}")
    else:
        model, config = load_from_checkpoint(args.checkpoint, DEVICE)
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
    for i in range(args.episodes):
        score = run_episode(agent, env_cfg)
        print(f"Episode {i + 1:2d}: {score:+.1f}")
        scores.append(score)
        env_cfg.seed = int(rng.integers(int(1e7)))

    print(f"\nMean over {args.episodes} episodes: {np.mean(scores):+.2f}")


if __name__ == "__main__":
    main()
