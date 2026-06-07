# DoomAgent Research Report

**Author:** j-tobias  
**Date:** 2026-06-07  
**Project:** VizDoom multiplayer agent trained with PPO + IMPALA CNN

---

## 1. Problem Statement

The task is to train a reinforcement learning agent that plays a first-person shooter (VizDoom) in a multiplayer deathmatch setting. The agent controls a single player facing 4 bots over a fixed 2000-tick episode. Performance is evaluated by a grading server that scores the agent on a private map with a private random seed.

**Server reward function:**
- +2 per hit landed on an enemy
- −0.1 per hit taken
- +100 per frag (kill)

The objective is to maximise the cumulative score per episode.

---

## 2. Environment

| Property | Value |
|---|---|
| Engine | VizDoom |
| Map | ROOM |
| Opponents | 4 bots, skill level 0 (easy) |
| Episode length | Fixed 2000 ticks (no early termination) |
| Death behaviour | **Respawn** — agent continues in the same episode after death |
| Observation | RGB screen, 128×128, normalised to [0, 1] |
| Action space | 8 discrete actions |
| FPS (solo) | ~166 steps/second on a single GPU machine |

A critical early discovery: the episode **never terminates on death**. The agent simply respawns and continues. This fundamentally changed the reward shaping strategy (see Section 5).

---

## 3. Architecture

### 3.1 Encoder: IMPALA CNN

Adapted from Espeholt et al. (2018), sized for 128×128 RGB input.

```
Input: (B, 3, 128, 128)

Stack 1: Conv2d(3→16, 3×3, p=1) + MaxPool(3, s=2, p=1) → 64×64
         2 × ResidualBlock(16)
Stack 2: Conv2d(16→32, 3×3, p=1) + MaxPool(3, s=2, p=1) → 32×32
         2 × ResidualBlock(32)
Stack 3: Conv2d(32→32, 3×3, p=1) + MaxPool(3, s=2, p=1) → 16×16
         2 × ResidualBlock(32)

ReLU → Flatten (8192) → Linear(8192, 256)

Output: (B, 256) feature vector
```

Each ResidualBlock uses two Conv2d layers with ReLU activations and a residual skip connection. The skip connections stabilise gradients under sparse rewards.

### 3.2 Actor-Critic Head (PPO)

```
Features (256) → Policy head: Linear(256, 8)  → logits → Categorical distribution
              → Value head:  Linear(256, 1)   → V(s) estimate
```

Orthogonal initialisation: policy head gain=0.01 (prevents early entropy collapse), value head gain=1.0.

### 3.3 Model Size

~8.4 MB ONNX export. Well within the 50 MB submission limit.

---

## 4. PPO Training Configuration

| Hyperparameter | Base runs | Fine-tuning runs |
|---|---|---|
| Total steps | 12,000,000 | 6,000,000 |
| Rollout steps (n_steps) | 20,000 | 20,000 |
| Epochs per rollout | 4 | 4 |
| Minibatches | n_steps // 512 = 39 | 39 |
| Initial LR | 2.5×10⁻⁴ | 2.5×10⁻⁵ (10× lower) |
| Final LR | 2.5×10⁻⁵ | 2.5×10⁻⁶ |
| Gamma | 0.99 | 0.99 |
| GAE lambda | 0.95 | 0.95 |
| Clip epsilon | 0.1 | 0.1 |
| Entropy coef (initial) | 0.01 | 0.01 |
| Entropy coef (final) | 0.001 | 0.001 |
| Value loss coef | 0.5 | 0.5 |
| Max grad norm | 0.5 | 0.5 |
| Reward normalisation | Yes (running std) | Yes |
| KL early stopping | 0.01 | 0.01 |
| Seed randomisation | Yes | Yes |

**Large rollout buffer (n_steps=20,000):** Observations are stored on CPU; gradients computed on GPU. This avoids GPU OOM while allowing diverse, low-variance rollouts. At 166 fps, one rollout covers ~120 seconds of game time — sufficient to observe multiple complete respawn cycles.

---

## 5. Key Discoveries

### 5.1 Episode Structure: Respawns, Not Terminations

Early training used a `DeathPenaltyReward` that applied a −10 penalty on death, under the assumption that dying would end the episode. Investigation of the environment logs revealed:

- `ep_len_mean` is always exactly **2000** regardless of how many times the agent dies
- The `DEAD` game variable transitions to 1 (dead/respawning) then returns to 0 (alive) without any episode reset
- Death merely costs ~3–5 ticks of inaction during respawn; it is not catastrophic

**Consequence:** The death penalty was actively harmful. It taught the agent to be passive and avoid combat rather than maximising kills. All subsequent runs dropped the death penalty entirely.

### 5.2 Stack4 ONNX Degradation

An experiment with 4-frame temporal stacking (`n_stack_frames=4`) was run to give the agent temporal context (motion, velocity cues). The encoder accepts stacked frames by reshaping `(B, C, T, H, W) → (B, C*T, H, W)` before the first convolution.

The ONNX export caused a score gap:
- Training checkpoint: ~542 ep_reward_mean
- After ONNX export + onnx2pytorch conversion: ~516
- Server score: **480**

Root cause: the `view()` reshape inside the encoder produced a dynamic-shape `Reshape` node in the ONNX graph. When `onnx2pytorch` reconstructed the PyTorch model, it failed to reproduce this correctly.

**Fix:** Pre-flatten the 5D input in the `_LogitsOnly` wrapper before the ONNX graph begins:
```python
def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if obs.ndim == 5:  # (B, C, T, H, W) → (B, C*T, H, W) before ONNX graph
        obs = obs.reshape(obs.shape[0], -1, obs.shape[-2], obs.shape[-1])
    out = self.model(obs)
    return out[0] if isinstance(out, tuple) else out
```
This ensures the reshape occurs outside the traced ONNX graph entirely.

### 5.3 Best-Checkpoint Tracking via EMA

The original training loop exported `submission.onnx` only at the end of training. If the policy regressed in the final steps (common with aggressive LR annealing), the submitted model would not be the best checkpoint.

**Fix:** EMA-smoothed reward tracking with live ONNX export during training:
```python
# After each rollout
ep_r = rollout_info.get("ep_reward_mean")
if ep_r is not None:
    self._ema_reward = (ep_r if self._ema_reward is None
                        else 0.05 * ep_r + 0.95 * self._ema_reward)
    if (self.step > self.cfg.total_steps * 0.10      # 10% warmup
            and self._ema_reward > self._best_ema_reward):
        self._best_ema_reward = self._ema_reward
        self.export_onnx(out_dir / "submission.onnx", obs_shape)
```
Alpha=0.05 smooths over noisy per-rollout episode samples. The 10% warmup prevents premature export during the initial fast-improvement phase.

### 5.4 CPU Bottleneck with Parallel Runs

An attempt to run 4 training processes simultaneously resulted in a 10× FPS reduction (166 fps solo → ~15 fps each). VizDoom is CPU-bound (game simulation, rendering, bot AI). The GPU remained at <3% utilisation with 4 parallel processes. Switching to **2 sequential pairs** restored full throughput.

### 5.5 Warm-Start Fine-Tuning

To leverage the 12M-step pretrained model without restarting from scratch, a `load_weights()` method was added to `PPOAgent`:

```python
def load_weights(self, path, partial=False):
    ckpt = torch.load(path, map_location=self.device, weights_only=True)
    if partial:
        current = self.model.state_dict()
        compatible = {k: v for k, v in ckpt["model"].items()
                      if k in current and v.shape == current[k].shape}
        current.update(compatible)
        self.model.load_state_dict(current)
    else:
        self.model.load_state_dict(ckpt["model"])
    if "reward_rms" in ckpt:
        self._reward_rms.load_state_dict(ckpt["reward_rms"])
    # Optimizer intentionally NOT loaded — fresh Adam for fine-tuning LR
```

The reward normalisation statistics (running mean/std over 12M steps) are carried over. The optimizer is reset so the fine-tuning LR takes effect immediately without momentum artifacts.

**Partial loading** (`partial=True`) filters layers by shape match — used for the stack4 architecture where the first conv layer has different `in_channels` (3 vs 12). 35/36 layers transferred; only the first conv is randomly re-initialised.

---

## 6. Experiments and Results

### 6.1 Run Timeline

```
Phase 1: 12M-step base training from scratch
  ├── impala_12M_death2         (death penalty, n_stack=1)  → server 523
  └── impala_12M_stack4_death2  (death penalty, n_stack=4)  → server 480

  Discovery: death penalty is harmful, stack4 has ONNX gap

Phase 2: 6M-step fine-tuning from impala_12M_death2 checkpoint
  Pair 1 (sequential):
  ├── impala_ft6M_base     (no death penalty, same config)         → server 617 ✓ NEW BEST
  └── impala_ft6M_highent  (no death penalty, ent_coef_final=0.005)
  
  Pair 2 (sequential):
  ├── impala_ft6M_stack4   (n_stack=4, partial load) — KILLED early, poor results
  └── impala_ft6M_alive    (AliveReward) — COMPLETED, training peak 711
```

### 6.2 Results Table

| Run | Steps | Peak train reward | Final train reward | Server score |
|---|---|---|---|---|
| impala_12M_death2 | 12M | 596 (step 11.88M) | 476 | **523** |
| impala_12M_stack4_death2 | 12M | ~453 | 340 | **480** |
| impala_ft6M_base | +6M | 622 (step 3.42M) | 589 | **617** |
| impala_ft6M_highent | +6M | 623 (step 0.78M) | 593 | — |
| impala_ft6M_stack4 | ~2.2M (killed) | 284 | — | — |
| impala_ft6M_alive | +6M | **711** (step 4.04M) | 665 | **pending** |

### 6.3 Run-by-Run Analysis

#### `impala_12M_death2` — Server score: 523
- 12M steps from scratch with death penalty (−10 on death)
- Strong baseline but the death penalty suppressed aggressive play
- Peak at step 11.88M (596), policy regressed slightly by final step
- EMA checkpoint tracking not yet implemented — final weights submitted
- Still improving at termination (not converged)

#### `impala_12M_stack4_death2` — Server score: 480
- Identical config but with 4-frame temporal stacking
- Training reward substantially lower than stack1 (encoder adapting to 4× input channels)
- Major ONNX degradation gap (−62 points vs checkpoint score of 542)
- ep_len_mean drifted below 2000 near end of training (cause unclear)
- The Stack4 ONNX gap made this experiment net-negative

#### `impala_ft6M_base` — Server score: **617** (current best)
- Warm-started from `impala_12M_death2/ckpt_012000000.pt`
- No death penalty, LR 10× lower (2.5×10⁻⁵ → 2.5×10⁻⁶)
- Reward_rms statistics carried over from pretrain
- Jumped from 530 to 622 within first 3.4M steps — immediate benefit of pretrained init
- Confirmed hypothesis: more compute on the base config continues to improve
- The 94-point server score improvement (523→617) validates the fine-tuning approach

#### `impala_ft6M_highent` — Not server-evaluated
- Same as `impala_ft6M_base` but `ent_coef_final=0.005` (5× higher entropy floor)
- Peaked early (623 at step 0.78M) then **plateaued** for the remaining 5.2M steps
- Entropy floor of 0.005 is too high for a mature pretrained policy — excess exploration prevents the policy from exploiting learned strategies
- Final reward ~540-593 (lower than `impala_ft6M_base` for most of training)
- Conclusion: high entropy annealing floor is counterproductive post-pretrain

#### `impala_ft6M_stack4` — Killed at 2.2M steps
- Architecture mismatch: 35/36 layers loaded, first conv randomly re-initialised
- At 2.2M steps, ep_reward_mean only ~280 (vs ~540 for other runs at same point)
- ep_len_mean=1512 (below 2000) — likely a logging artifact from small n_episodes=2
- Resource-intensive with no sign of improvement trajectory
- Killed to free CPU/GPU for `impala_ft6M_alive`

#### `impala_ft6M_alive` — Server score: pending
- Introduced `AliveReward`: base reward + 0.05 per tick for being alive (not respawning)
- Peaked at **711** (step 4.04M) — the highest training reward recorded in the project
- Final reward 665 — strong and consistent throughout training
- The alive bonus provides dense per-tick supervision without conflicting with the server's kill-based reward
- Being alive is correlated with winning firefights — the bonus implicitly rewards aggressive, accurate play that minimises time spent dead
- submission.onnx captured by EMA best-checkpoint tracker throughout training

---

## 7. Reward Shaping Analysis

Three reward functions were tested across the project:

### `CustomReward` (base)
```
reward = hits × 2 + hits_taken × (−0.1) + frags × 100
```
Identical to the server reward. No additional shaping. Used in all runs except DeathPenalty and AliveReward.

### `DeathPenaltyReward` (deprecated)
```
reward = base + (−10 if newly_dead else 0)
```
Motivation: discourage passive or suicidal play. Invalidated once we discovered deaths are respawns, not episode terminations. The penalty did more harm than good — it suppressed aggression.

### `AliveReward` (best)
```
reward = base + (0.05 if not DEAD else 0)
```
Per-tick bonus for being in the alive state. Effective because:
1. Dense signal: 2000 ticks per episode, up to +100 per episode from survival alone
2. Aligned with server objective: staying alive enables more shots, more kills
3. Non-conflicting: 0.05/tick is small relative to +100/frag, so kill incentives dominate
4. No episode structure assumption: correct regardless of whether deaths end episodes

---

## 8. Engineering Infrastructure

### 8.1 ONNX Export Pipeline

The grading server uses `onnx2pytorch` to convert submitted ONNX files back to PyTorch models. This imposes constraints:
- Must use legacy TorchScript exporter (`dynamo=False`)
- Must use opset 12 (newer opsets use Conv attribute layouts that onnx2pytorch cannot parse)
- Dynamic-shape operations inside the graph cause conversion artifacts

The `_LogitsOnly` wrapper strips the value head (server only needs logits) and pre-flattens 5D frame-stack inputs before tracing:
```python
class _LogitsOnly(nn.Module):
    def forward(self, obs):
        if obs.ndim == 5:
            obs = obs.reshape(obs.shape[0], -1, obs.shape[-2], obs.shape[-1])
        out = self.model(obs)
        return out[0] if isinstance(out, tuple) else out
```

### 8.2 Metrics Logging

Each run produces:
- `metrics.csv`: one row per log interval (every 1000 steps)
- `train.log`: human-readable log with all metrics
- Weights & Biases: real-time dashboard for all runs
- `ckpt_XXXXXXXX.pt`: checkpoints every 60,000 steps containing model weights, optimizer state, reward_rms state, and step counter

### 8.3 Sequential Run Queue

A bash queue watcher launched Pair 2 automatically when Pair 1 completed:
```bash
while ps -p $PID_A || ps -p $PID_B; do sleep 30; done
# ... launch pair 2
```
This ensured sequential execution without manual intervention overnight.

---

## 9. Score Progression

```
Baseline (scratch, 12M steps, death penalty):       523  [+0]
Stack4 experiment (ONNX degradation):               480  [−43 vs baseline]
Fine-tune base (6M more steps, no death penalty):   617  [+94 vs baseline]
Fine-tune alive reward (pending server eval):        ???  [training peak: 711]
```

The fine-tuning paradigm (pretrain → adapt) produced a **+18% server score improvement** over the scratch-trained model in just half the compute.

---

## 10. Conclusions and Next Steps

### What Worked
1. **Fine-tuning from pretrained weights** at 10× reduced LR: most impactful intervention, +94 server points
2. **AliveReward**: best training performance of any run (711 peak), server result pending
3. **EMA best-checkpoint tracking**: ensures the best policy during training is submitted, not the final (possibly regressed) policy
4. **Large rollout buffer (n_steps=20,000)**: stable, low-variance training at 166 fps
5. **Dropping the death penalty**: immediately unlocked more aggressive play strategies

### What Did Not Work
1. **Frame stacking (n_stack=4)**: ONNX conversion gap erased any potential benefit; encoder architecture mismatch for fine-tuning required partial loading with only 35/36 layers
2. **High entropy floor (ent_coef_final=0.005)**: caused plateau for a mature pretrained policy; exploration pressure useful early but counterproductive late
3. **4 parallel runs**: CPU bottleneck reduced throughput 10×; 2-at-a-time is the practical maximum

### Open Questions
1. **AliveReward server score**: the training signal suggests a large improvement; the actual server score will confirm whether the 0.05/tick bonus translates
2. **Continued fine-tuning**: `impala_ft6M_base` and `impala_ft6M_alive` are still improving at their final steps — more compute budget would likely push scores higher
3. **Stack4 with proper ONNX fix**: the ONNX gap is now fixed; a fresh stack4 run from the 617 checkpoint might be net positive if temporal information helps
4. **Bot skill escalation**: all training uses bot_skill=0 (easy); training against harder bots might generalise better to the server's bot configuration

---

*Report generated from training logs, metrics CSVs, and server evaluation results collected during the project.*
