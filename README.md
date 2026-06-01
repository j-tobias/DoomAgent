# DoomAgent

VizDoom deathmatch agent — JKU Deep RL 2025.

## Setup

```bash
git clone --recurse-submodules <repo>
uv sync
echo "$PWD/jku.wad" > .venv/lib/python3.13/site-packages/jku_wad.pth
```

## Usage

```bash
# Train
uv run scripts/train.py --run-name my_run

# Evaluate a submission
uv run scripts/evaluate.py --submission runs/my_run/submission.onnx
```
