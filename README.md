# latent-teleop

Latency-robust teleoperation with a latent world model. This repo currently covers **milestone G1**: can an
action-conditioned predictor on frozen DINOv2 features predict a contact-rich scene far enough ahead to be worth
showing an operator? Task: PushT (gym-pusht, pymunk physics) at 30 ms ticks, driven by a scripted operator
through a simulated delayed network.

## Files

| file | what it does |
|---|---|
| `sim.py` | PushT at 30 ms ticks; full state, render-from-state, contact test, exact agent PD replay, scripted operator |
| `teleop.py` | delayed uplink/downlink (`Link`), delay profiles incl. bursts and `trace:<file>` replay, the teleop loop |
| `collect.py` | records episodes per delay profile (states only, frames are re-rendered later) |
| `embed.py` | renders every state at 224 px, encodes with frozen DINOv2-S, saves CLS + pooled patch tokens |
| `world_model.py` | C1 predictor (block-causal transformer, ordered action tokens, KV-cached rollout) + probe |
| `train.py` | trains probe only (`--probe-only`) or predictor + probe with teacher-forcing + 20-step rollout loss |
| `eval_g1.py` | error vs horizon, learned vs hold / constant-velocity / kinematic, writes the G1 verdict and chart |
| `probe_check.py` | perception-floor diagnostics (MLP vs spatial probe vs raw pixels) |
| `smoke.sh` | end-to-end run on tiny data; must print `SMOKE OK` |
| `fleet.sh` | push code to servers, build the venv, run the self-checks: `./fleet.sh 91 70 55` |
| `prototype/` | the original 400-line prototype, kept for reference (superseded) |

Every non-trivial module has a self-check: `python sim.py`, `python teleop.py`, `python world_model.py`.

## Design decisions

**Where delay enters.** The predictor is a dynamics model in *robot time* (frame t + command executed during tick
t -> frame t+1). It does not take the delay as an input. Delay enters through *which* actions are fed in and *how
many* steps are rolled out. The operator holds a frame captured at robot tick τ. Frames will carry the id of the
last command the robot executed (C2). The commands still in flight are placed on the robot's timeline using the
measured uplink delay, and the model rolls forward to the tick the next command will land. This is how
"condition on the measured one-way delay" from the proposal is implemented. G1 feeds the *true* executed
commands, so it measures the model's upper bound. Errors in estimating the delay are a C2 question.

**The agent is exact and the block is the question.** The PushT agent is a kinematic body, so replaying its PD
controller on the executed commands reproduces it exactly (checked in `sim.py`). The ghost will draw the agent
kinematically. The learned model only has to earn its place on the **block pose**.

**The block is quasi-static** (`damping = 0`): it only moves while it is being pushed. In free space,
hold-last-frame is perfect for the block, so every metric is split by phase.

**Latency layer fixes vs prototype.** The screen holds the last delivered frame and never leaks the live one.
A command that arrives after a newer one is dropped as stale. When no command arrives, the robot holds its last
target, which is the stall that C3 will address. All of these are asserted in `teleop.py`.

## G1 gate (frozen before any full-data numbers)

Measured on the 180-episode diagnostic set: in windows that contain contact, the block moves a median of only
2.7 px in 150 ms and 5.4 px in 300 ms (mean 5.4 / 9.4 px). A 150 ms gate would therefore be decided by
perception noise, so the gate uses longer horizons and only windows where the block actually moves.

G1 **PASSES** if all of these hold on the test episodes (every 10th episode):
1. Perception floor (probe on true frames): block position error < 3 px and block angle error < 3°.
2. In *block-moving* windows (the block moves ≥ 5 px or rotates ≥ 3° between the anchor and the predicted tick), at
   **300 ms and 600 ms**: the learned model's block position error **and** block angle error are both ≤ 0.8 × the
   best of hold-last-frame, constant velocity and kinematic. The baselines read the *true* delayed state, which
   makes this the strict comparison.

If G1 fails, the next phase narrows (for example, the ghost shows the block only within the horizon where it
beats hold) rather than stopping.

## Status (2026-10-21)

Verified:
- Self-checks pass on 7 servers: .91, .70, .55, .87 (RTX 3050, bf16) and .54, .63, .100 (GTX 1660 Super, fp32).
  Not used: .90 (another user at 100% GPU), .89 (shared desktop), .51/.93 (NVIDIA driver not loaded),
  .73 (no GPU), .86/.88/.72/.92 (SSH times out).
- Smoke test passes end to end on .91 (about 1 min).
- Seeded collection is bit-identical across machines (states, commands, timing). Only shapely's coverage
  float differs, by 3e-16.
- Scripted operator at 0 ms: 20% of ticks in contact. It brings the block from 130 px to 33 px from the goal
  and from 97° to 12° of rotation, but only 10% of episodes reach PushT's 95% coverage within 18 s. That is
  enough contact-rich data for G1. It is too weak for RQ2 completion-time studies (tune it before C2).
- Perception floor (`probe_check.py`, 180-episode diagnostic set): pooling resolution was not the limit and neither
  was a pipeline bug. The probe reaches 2.6 px / 0.9° on its *training* frames. It generalises poorly because
  there is too little pose coverage. The spatial 8×8 probe's validation block error falls with data:
  12.4 px (36 episodes) -> 8.9 (72) -> 6.4 (144), about 28% per doubling. 8×8 ≈ 16×16, and both are far better
  than 4×4 or CLS-only. **Chosen: 8×8 pooling (65 tokens) + spatial probe.** Extrapolating to the full run's ~960
  training episodes gives about 2.6 px / 1.3°, just under gate condition 1. *That is an extrapolation
  from three points, so condition 1 can still fail.* The floor is printed to `data/probe.log` about 40 minutes into
  the run. The run does not stop on a bad floor, because condition 2 still tells us where to narrow, but you can
  kill it there. The floor, the gate and the rollout decoding all use this one probe, trained on true frames only.
  (The 56×56 raw-pixel CNN sanity probe failed at about 30–45 px. That is an issue in that CNN, not in the data:
  the renders and targets were checked by hand, and the token probes fit the training frames to 2.6 px.)
- Predictor cost at 65 tokens/frame on an RTX 3050: 0.87 s/step at batch 16, peak 5.0 GB (batch 32 runs out of
  memory). The KV-cached rollout was needed: recomputing every tick took 3.3 s per step at batch 64 even with
  17 tokens.

Known limits carried into C2:
- The trained horizon is K = 20 ticks = 600 ms. A ghost under the 300 ms profiles needs downlink age plus uplink
  delay, about 600 ms on average and more during bursts, so C2 needs a larger K. That is a training-cost question.
- The agent reads back from the latent at only ~12 px on validation. That does not matter, because the ghost draws
  the agent kinematically (exactly).
- Deviations from the proposal: synthetic delay profiles (real WAN traces plug in as `trace:<file>`); delay
  conditioning via the action timeline instead of as a network input; DINOv2-S instead of V-JEPA 2 (too slow on
  6 GB cards); C2–C4 not built yet.

## Full G1 run (not started; waiting for the go)

| stage | time on one RTX 3050 | output |
|---|---|---|
| collect 1200 episodes (6 delay profiles × 200) | ~1 min (CPU) | `data/episodes/` (~0.3 GB) |
| embed ~708k frames, DINOv2-S, 8×8 | ~20 min | `data/emb_p8.npy` (~35 GB, memory-mapped) |
| probe-only floor check | ~15 min | `data/probe.log`: the floor number appears here |
| predictor + probe, 40k steps × batch 16 | ~10 h | `data/wm_<tag>.pt`, `data/train_<tag>.log` |
| G1 eval | ~10 min | `results/<tag>/g1.png`, `g1_metrics.csv`, `g1_verdict.json` |

Plan: the main run on .91 and the no-rollout-loss ablation on .70, in parallel (each builds its own identical data):

```bash
ssh yashdewangan@10.81.30.91 'cd ~/latent-teleop && nohup ./run_g1.sh main > run_main.log 2>&1 &'
ssh yashdewangan@10.81.30.70 'cd ~/latent-teleop && nohup ./run_g1.sh no_rollout --rollout-weight 0 > run_no_rollout.log 2>&1 &'
```

## Running

```bash
./fleet.sh 91                          # sync + venv + self-checks
ssh yashdewangan@10.81.30.91 'cd ~/latent-teleop && ./smoke.sh'
```
