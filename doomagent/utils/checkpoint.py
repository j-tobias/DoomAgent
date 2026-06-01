from pathlib import Path

import torch


def save_checkpoint(agent, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": agent.step,
        "model": agent.model.state_dict(),
    }
    if hasattr(agent, "optimizer"):
        state["optimizer"] = agent.optimizer.state_dict()
    torch.save(state, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(agent, path: Path) -> None:
    ckpt = torch.load(path, map_location=agent.device, weights_only=True)
    agent.step = ckpt["step"]
    agent.model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt and hasattr(agent, "optimizer"):
        agent.optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Checkpoint loaded ← {path}  (step {agent.step})")
