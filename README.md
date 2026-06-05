<p align="center">
  <img src="resources/DoomAgentTitle.png" alt="DoomAgent" width="100%">
</p>

<div align="center">

[Overview](#overview) · [Results](#results) · [Architecture](#architecture) · [Training](#training) · [Evaluate](#evaluate)

</div>

---

## Overview

A PPO-based agent for the [JKU Deep Reinforcement Learning](https://www.jku.at/) course challenge: survive and dominate a VizDoom deathmatch against 4 bots on the ROOM map. Submitted as an ONNX model (≤ 50 MB) evaluated by the grading server.

**Task setup:**
- 1 player vs 4 bots · ROOM map · `episode_timeout = 2000`
- `Discrete(8)` action space · `128×128` RGB observations
- Reward: `+2` per hit · `-0.1` per hit taken · `+100` per frag

---

## Recorded Run

<video src="resources/recorded_run.mp4" controls width="100%"></video>

*Best episode from `impala_6M` — +577.8 score (~5.8 net frags)*

---

## Results

| Run | Steps | Server score | Key changes |
|---|---|---|---|
| `ppo_baseline` | 1M | — | NatureCNN, no extras |
| `impala_2M` | 2M | — | IMPALA encoder |
| `impala_4M_big` | 4M | — | + large rollout buffer (20k steps) |
| `impala_6M` | 6M | **490** ✓ | + reward norm, entropy anneal, KL stop |
| `impala_10M_death` | 10M | 483 | + death penalty (−10) |
| `impala_10M_stack4` | 10M | 420 | + frame stacking (n=4) |

**Best submission:** `impala_6M` · server score **490**

---

## Architecture

**Encoder — IMPALA CNN** (2.2M params, 256-dim output)

```
Conv(in_ch → 16, 3×3) + MaxPool  →  64×64
  2 × ResBlock(16)
Conv(16 → 32, 3×3)   + MaxPool  →  32×32
  2 × ResBlock(32)
Conv(32 → 32, 3×3)   + MaxPool  →  16×16
  2 × ResBlock(32)
ReLU → Flatten → Linear(8192, 256)
```

**Actor-Critic heads** on top of the encoder, orthogonal init (policy gain=0.01, value gain=1.0).

**Training stabilisers:**
- Reward normalisation via Welford's online `RunningMeanStd`
- Entropy coefficient annealing `0.01 → 0.001`
- Value function clipping
- KL early stopping (`target_kl=0.01`)
- LR annealing with 10% floor
- Large CPU rollout buffer (20k steps) — stored on CPU, batched to GPU

---

## Training

```bash
# Reproduce impala_6M (best submission)
uv run scripts/train.py \
  --run-name impala_6M \
  --total-steps 6_000_000 \
  --n-steps 20000

# With frame stacking
uv run scripts/train.py \
  --run-name my_run \
  --total-steps 10_000_000 \
  --n-steps 4096 \
  --n-stack-frames 4

# With death penalty
uv run scripts/train.py \
  --run-name my_run \
  --total-steps 10_000_000 \
  --death-penalty \
  --death-penalty-value 2
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--total-steps` | 1M | Total environment steps |
| `--n-steps` | 512 | Rollout buffer size |
| `--n-stack-frames` | 1 | Frame stacking depth |
| `--encoder` | `impala` | `impala` or `nature` |
| `--death-penalty` | off | Add penalty on death |
| `--death-penalty-value` | 10.0 | Penalty magnitude |
| `--no-reward-norm` | — | Disable reward normalisation |
| `--no-ent-anneal` | — | Disable entropy annealing |
| `--no-random-seeds` | — | Fix VizDoom spawn seed |

---

## Evaluate

```bash
# Clone with submodule and install
git clone --recurse-submodules <repo>
uv sync
echo "$PWD/jku.wad" > .venv/lib/python3.13/site-packages/jku_wad.pth

# Local evaluation (10 episodes)
uv run scripts/evaluate.py --checkpoint runs/<run>/ckpt_XXXXXXXXX.pt

# Server-equivalent evaluation (uses onnx2pytorch)
uv run resources/server_eval_doom.py --submission runs/<run>/submission.onnx

# Record a video of the best episode
uv run scripts/record.py --checkpoint runs/<run>/ckpt_XXXXXXXXX.pt --record-best
```

---

## Project Structure

```
DoomAgent/
├── src/doomagent/
│   ├── agents/          # PPOAgent, DQNAgent
│   ├── buffers/         # RolloutBuffer (PPO), ReplayBuffer (DQN)
│   ├── models/          # PPOActorCritic, DQNModel, IMPALAEncoder, NatureCNN
│   ├── utils/           # Logger, RunningMeanStd, ONNX export
│   ├── config.py        # PPOConfig, DQNConfig, EnvConfig
│   ├── env.py           # make_env()
│   └── reward.py        # CustomReward, DeathPenaltyReward
├── scripts/
│   ├── train.py         # PPO training entry point
│   ├── evaluate.py      # Local evaluation
│   └── record.py        # Video recording
├── resources/
│   ├── recorded_run.mp4
│   └── server_eval_doom.py
├── jku.wad/             # Git submodule — VizDoom environment
└── Note.md              # Experiment notes and next steps
```
