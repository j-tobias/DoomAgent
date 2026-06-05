# Experiment Notes

## Server score leaderboard

| Run | Steps | Server | Local eval | Notes |
|---|---|---|---|---|
| `impala_6M` | 6M | **490** | 488.6 | Current best submission |
| `impala_10M_death` | 10M | 483 | 397.5 | Death penalty -10 |
| `impala_10M_stack4` | 10M | 420 | 512.2 | Frame stacking (n=4) |

---

## Learnings from impala_10M runs

### Local eval is unreliable without seed randomisation
Our local eval used a fixed seed (1337) that happened to be unlucky for the death run (+397 local vs +483 server) and lucky for the stack4 run (+512 local vs +420 server). The `impala_6M` was consistent (+488.6 vs +490) because it was evaluated over more seeds during development. **Fix:** always run the server eval script for final numbers; local eval is only useful for relative comparisons within a session.

### Death penalty -10 is too strong
The `-10` on death caused the agent to converge to cautious/passive behaviour (very low entropy 0.65 at end, `ep_len=2000` always — never dying but also scoring fewer frags). Despite 10M steps vs 6M, it only reached 483 vs 490. Reducing to `-2` (same magnitude as a hit reward) should preserve aggression while discouraging unnecessary deaths.

### Frame stacking works locally but needs robustness
Stack4 showed strong late-training rewards (~450–500 in the final rollouts) and a promising local server eval (+512). On the actual grading server it dropped to 420. Two likely causes:
1. **Seed sensitivity** — training on a single fixed seed (1337) means the model may have overfit to specific spawn positions. A different seed set exposes this.
2. **Potential onnx2pytorch reshape issue** — the 5D→4D `view()` op in the ONNX graph is more complex than anything in previous models. onnx2pytorch converted it correctly in our tests, but behaviour may differ on the server's onnx2pytorch version.

### Training on a single seed is a critical weakness
All runs so far trained on seed=1337 exclusively. The grading server uses its own fixed seed set, meaning performance depends heavily on whether those specific spawn configurations were seen during training. This is likely a bigger factor than architecture or reward shaping.

---

## Code improvements made

| Feature | Where | Detail |
|---|---|---|
| Frame stacking | `encoder.py` | 5D→4D reshape guard in `forward()` of both encoders |
| Frame stacking CLI | `train.py` | `--n-stack-frames N` arg; `in_channels = base_ch * N` |
| Checkpoint eval fix | `evaluate.py` | Infers `n_stack_frames` from first conv weight shape |
| Configurable death penalty | `reward.py` | `DeathPenaltyReward(death_penalty=N)` |
| Death penalty CLI | `train.py` | `--death-penalty-value N` (default 10) |
| Seed randomisation | `agents/ppo.py`, `env.py` | `reseed_env()` called on every episode reset |
| Seed randomisation CLI | `train.py` | `--no-random-seeds` to disable; on by default |

---

## Next run: `impala_12M_stack4_death2`

Combines all learnings into one run.

```bash
CUDA_VISIBLE_DEVICES=0 nohup uv run scripts/train.py \
  --run-name impala_12M_stack4_death2 \
  --total-steps 12_000_000 \
  --n-steps 4096 \
  --n-stack-frames 4 \
  --death-penalty \
  --death-penalty-value 2 \
  > runs/impala_12M_stack4_death2/train.log 2>&1 &
```

**Why each knob:**
- `n-stack-frames 4` — gives the model motion information for aiming
- `death-penalty-value 2` — mild deterrent (same scale as a hit reward), keeps aggression intact
- `random-seeds` (default on) — exposes diverse spawn positions during training, reducing seed sensitivity on the grading server
- `n-steps 4096` — required to keep the CPU rollout buffer under ~3.2 GB with 4-frame obs
- `12M steps` — death run still improving at 10M; extra 2M gives more room to converge
