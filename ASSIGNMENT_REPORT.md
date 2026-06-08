# Assignment V: Learning to Fight with Deep Reinforcement Learning

**Justus Tobias** **K12102675** · Deep Reinforcement Learning, JKU · June 2026

---

## Task & Architecture

**Environment.** VizDoom deathmatch — a first-person shooter with a 128×128 RGB observation and 8 discrete actions (move forward/backward, strafe left/right, turn left/right, shoot, do nothing). The agent faces 4 bots over a fixed 2000-tick episode; the agent respawns on death and the episode never terminates early. The server reward is +2 per hit landed, −0.1 per hit taken, and +100 per frag (kill).

**Preprocessing.** The RGB screen is normalised to [0, 1]; no further spatial cropping or resizing. A single frame is fed per step (no temporal stacking in the best submission). All runs used seed randomisation during training — the environment is reseeded at every episode reset to expose diverse spawn configurations and reduce overfitting to fixed initial states.

**Network.** An IMPALA CNN encoder followed by a two-headed actor-critic.

| Layer | Type | Output shape |
| --- | --- | --- |
| Stack 1 | Conv2d(3→16, 3×3, p=1) + MaxPool(3, s=2, p=1) + 2×ResBlock(16) | 16×64×64 |
| Stack 2 | Conv2d(16→32, 3×3, p=1) + MaxPool(3, s=2, p=1) + 2×ResBlock(32) | 32×32×32 |
| Stack 3 | Conv2d(32→32, 3×3, p=1) + MaxPool(3, s=2, p=1) + 2×ResBlock(32) | 32×16×16 |
| Flatten | — | 8 192 |
| FC | Linear(8192→256) + ReLU | 256 |
| Policy head | Linear(256→8) | 8 logits |
| Value head | Linear(256→1) | V(s) |

Each ResidualBlock contains two Conv2d layers with a skip connection. Orthogonal initialisation is used throughout: policy head gain=0.01 (prevents early entropy collapse), value head gain=1.0. Model size: ~8.4 MB ONNX export, well within the 50 MB limit.

**Algorithm.** Proximal Policy Optimisation (PPO) with GAE advantage estimation. Rollouts of 20,000 steps are collected on CPU; gradient updates run on GPU over 4 epochs per rollout with minibatch size 512. The learning rate is linearly annealed and entropy coefficient decays from 0.01 → 0.001 to balance exploration and exploitation. KL early stopping (threshold 0.01) prevents destructive policy updates.

## Hyperparameter Search & Findings

Base runs used 12M steps from scratch at LR 2.5×10⁻⁴; fine-tuning runs warm-started from the best base checkpoint at 10× lower LR (2.5×10⁻⁵):

| Run | Steps | LR | Death penalty | Stack | Reward shaping | Server score |
| --- | --- | --- | --- | --- | --- | --- |
| impala_12M_death2 | 12M | 2.5×10⁻⁴ | −10 | 1 | Base | 523 |
| impala_12M_stack4_death2 | 12M | 2.5×10⁻⁴ | −10 | 4 | Base | 480 |
| **impala_ft6M_base** | **+6M** | **2.5×10⁻⁵** | **none** | **1** | **Base** | **617** |
| impala_ft6M_highent | +6M | 2.5×10⁻⁵ | none | 1 | Base (ent_final=0.005) | — |
| impala_ft6M_stack4 | ~2.2M (killed) | 2.5×10⁻⁵ | none | 4 | Base | — |
| impala_ft6M_alive | +6M | 2.5×10⁻⁵ | none | 1 | +0.05/tick alive | 586 |

Key findings:

1. **Deaths are respawns, not terminations.** The episode is always exactly 2000 ticks; the agent respawns after death with ~3–5 ticks of inaction. The death penalty (−10) therefore penalised aggression rather than careless play, teaching the agent to avoid combat. Removing it immediately unlocked more aggressive kill-seeking behaviour. All subsequent runs dropped the penalty entirely.

2. **Fine-tuning from a pretrained checkpoint is the most impactful intervention.** Warm-starting from `impala_12M_death2` at 10× lower LR (2.5×10⁻⁵) with no death penalty raised the server score from 523 → 617 (+18%) in just half the compute (6M vs 12M steps). The pretrained reward normalisation statistics were carried over; the optimizer was reset so the fine-tuning LR took effect without momentum artifacts.

3. **AliveReward inflated training metrics without improving real performance.** A +0.05/tick bonus for being alive peaked at a training reward of 711 — the highest in the project — but scored only 586 on the server. The bonus contributes ~95 pts/episode (2000 ticks × 0.05), which the server does not award. Stripping it yields 711 − 95 ≈ 616, nearly identical to the base fine-tune run. The bonus also introduced mild risk-aversion: the agent avoided situations where dying led to respawn ticks, even when aggression would have produced more frags.

4. **Frame stacking (n=4) lost value through ONNX conversion.** A dynamic `view()` reshape inside the encoder produced an ill-formed `Reshape` node in the ONNX graph that `onnx2pytorch` could not reconstruct correctly, causing a gap of ~62 points between the training checkpoint and the server score. Fix: pre-flatten the 5D frame-stack input in the `_LogitsOnly` export wrapper, outside the traced graph. The fine-tuning attempt with the fixed stack4 architecture was abandoned after 2.2M steps due to poor early reward and resource pressure.
