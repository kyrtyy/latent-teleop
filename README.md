# aktoria_teleop

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
| `ghost.py` | C2 ghost: delay estimate from execution stamps, command timeline, none / kinematic / learned / oracle |
| `c2.py` | C2 closed-loop experiment, paired seeds, batched learned ghost |
| `c2_gate.py` | C2 gate: gap recovered, paired-bootstrap CIs, verdict |
| `c3.py` | C3: path operator, chunk executor with blending, world-model / oracle safety layer |
| `c4.py` | C4: fault injection, segment scores, conformal flags |
| `policy.py` | RQ4: state-based diffusion policy, curation arms, autonomous evaluation |
| `leak_check.py` | re-runs the G1 rollout with corrupted future commands, per delay profile |
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

## Status before the full run (2026-10-21)

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

## G1 results (2026-10-22)

Main run (.55, 40k steps, ~10 h) and the no-rollout-loss ablation (.70). Test split, block-moving windows,
learned model vs the best baseline reading the *true* state (`results/<tag>/g1_verdict.json`):

| | main | ablation (`--rollout-weight 0`) | best baseline |
|---|---|---|---|
| block px, 300 ms | **6.9** (0.65x) | 12.8 (1.21x) | 10.6 (const-vel) |
| block deg, 300 ms | **3.2** (0.40x) | 7.7 (0.95x) | 8.1 (const-vel) |
| block px, 600 ms | **7.2** (0.40x) | 17.8 (1.00x) | 17.9 (hold) |
| block deg, 600 ms | **4.0** (0.28x) | 11.4 (0.80x) | 14.2 (hold) |

**With the original probe, G1 fails only on condition 1** (perception floor 4.7 px / 1.8°, gate < 3 px). Condition 2 passes on all four
cells, with margin. The rollout loss is what makes the predictor useful: the teacher-forced ablation is no better
than hold. Of the learned 7.2 px, about 4.7 px is the probe floor. In free space hold is exact and the learned
model sits at the floor, so the ghost should show the learned block only when contact is expected.

**Leak check** (`leak_check.py`): corrupt the future commands and the win disappears (600 ms block: true commands
7.2 px, last command held 14.7, commands from another window 17.3, hold-last-frame 17.9). The win is
action-driven dynamics, not future frames. It also holds under 300 ms delay (7.4 vs 14.2 px), where the window's
commands were issued from pre-anchor frames, so the commands cannot carry the operator's reaction to the future
block.

**Perception floor fix → G1 PASS** (`floor.sh`, 2026-10-22). Only the probe is retrained, on true frames from 4×
the episodes (4,800; episodes 0–1199 are bit-identical to `data/`, so the test split is unchanged and unseen).
The predictor is the same `wm_main.pt`. Results are scored on the original test set:

| probe | floor block px / deg | block px / deg @ 600 ms, moving (ratio) | G1 |
|---|---|---|---|
| 1,200 eps, 20k steps | 4.71 / 1.84 | 7.2 / 4.0 (0.40 / 0.28) | FAIL (floor) |
| 4,800 eps, 60k steps | 3.27 / 1.18 | 6.3 / 3.1 (0.35 / 0.22) | FAIL (floor) |
| 4,800 eps, 150k steps | **2.97 / 1.13** | **6.2 / 2.8 (0.34 / 0.20)** | **PASS** |

Caveat: condition 1 passes by 0.03 px, with one seed and on the third probe tried. Condition 2 is the robust result
(every cell ≤ 0.54×). The data-scaling rate for the floor dropped (4× data gave only −30%), so the next gain is a
finer readout (unpooled 16×16 tokens), not more episodes.

`train.py` saves `<out>.resume` every eval and resumes from it. The first main run on .91 was killed at
step 26k (no traceback, a desktop user had logged in) and was rerun on .55.

## C2: ghost display (closed loop)

`ghost.py` builds what the operator perceives from what it can actually observe: received frames (stamped with
the robot's capture tick and the command it executed the tick before) and its own sent commands. It estimates the
age of executed commands from the stamps, rebuilds the robot's command timeline, and shows the scene at
T = now + estimated delay (the tick the command being sent now starts to act). Kinds: `none` (delayed frame),
`kinematic` (agent replayed exactly through its PD, block held: a Smith predictor), `learned` (kinematic agent +
world-model block, used when the agent's predicted path reaches the block), `oracle` (true physics from the full
state at the frame, on the same estimated timeline). Past 600 ms the rollout is chained: at 1.5 s the block
error is still 7.0 px vs 26.7 px for hold (`results/main_long`), so no retraining for longer horizons.

Checks: zero delay -> every ghost is the live frame; constant delay -> the estimate is exact and the kinematic agent
is exact; 300b -> estimated age within 1 tick of the truth, agent 6.5 px vs 69 px from the delayed frame. The
refactored teleop loop reproduces recorded training episodes bit for bit.

`c2.py`: scripted operator, 30 s episodes, N = 200 paired seeds per cell (same scene, operator parameters and
delay sequence in every cell; seeds disjoint from the training data). The learned ghost runs all episodes in
lockstep with one batched encode and one batched rollout per tick.

### C2 gate (frozen before any C2 numbers)

Metric: coverage AUC, the mean PushT coverage over the 30 s episode (held at its final value after success).
Gap recovered = (AUC_ghost - AUC_none) / (AUC_0ms - AUC_none), per profile. CIs are paired bootstrap, 95%.

C2 **PASSES** if on both 300j and 300b:
1. the learned ghost helps: AUC_learned - AUC_none > 0, CI excluding 0; and
2. the world model adds something beyond kinematics: AUC_learned - AUC_kinematic > 0, CI excluding 0.

Reported regardless: gap recovered for every ghost and profile, the oracle as the ceiling, success rate, and the
closed-loop ghost error (the operator behaves differently with a ghost, so this is not the G1 error). If (1) holds
and (2) does not, the finding is that a Smith predictor is enough for this task, and the learned model is not
needed for the display.

### C2 results (2026-10-23): **PASS**

`python c2_gate.py` on `results/c2` (N = 200 paired episodes per cell, model `data/wm_main.pt` retrained on .91,
probe `data_big/probe150k_p8.pt`). Coverage AUC, with gap recovered in parentheses. The 0 ms reference is 0.516
(success 0.24).

| Profile | none | kinematic | learned | oracle | learned − kinematic [95% CI] |
|---|---|---|---|---|---|
| 50j | 0.457 | 0.490 (56%) | 0.498 (70%) | 0.517 (103%) | +0.008 [−0.008, +0.024] |
| 150j | 0.375 | 0.451 (54%) | 0.514 (99%) | 0.521 (104%) | +0.063 [+0.046, +0.078] |
| 150b | 0.363 | 0.438 (49%) | 0.504 (93%) | 0.516 (100%) | +0.066 [+0.052, +0.081] |
| **300j** | 0.298 | 0.426 (59%) | **0.494 (90%)** | 0.516 (100%) | **+0.068 [+0.051, +0.084]** |
| **300b** | 0.287 | 0.419 (58%) | **0.483 (86%)** | 0.507 (96%) | **+0.064 [+0.048, +0.080]** |

At 300j and 300b, learned − none is +0.196 (CI [+0.18, +0.21]), and both gate conditions hold.

| 300 ms (j / b) | none | kinematic | learned | oracle |
|---|---|---|---|---|
| success rate | 0.04 / 0.03 | 0.14 / 0.12 | 0.20 / 0.22 | 0.33 / 0.31 |
| closed-loop ghost block error while moving (px) | 22.0 / 22.3 | 24.2 / 24.0 | 7.7 / 8.8 | 3.7 / 4.8 |

The learned ghost queries the model about 540 times per 30 s episode and falls back to hold 0.9 (300j) and 11.3
(300b) times per episode. The delay cap never triggers.
At 50 ms there is too little delay for the model to beat kinematics; the CI includes 0. 300 ms cells ran with
rollout chunk 16, because 64 ran out of memory on a 6 GB GPU.

## C3: intent-preserving execution

`c3.py`: every tick the operator sends its intended path, the next 10 targets it would send if the scene stayed as
it now looks, instead of one setpoint. The robot keeps running the newest path while packets are late, and after a
gap it blends into the new path over 5 ticks (real-time-chunking style). An optional safety layer runs only while
the robot is extrapolating on its own. It stops if following the rest of the path would move the block more than
holding still would (3 px or 0.05 rad over 10 ticks). `wm` judges this with the C1 world model on the robot's own
live frames, differentially, so the probe's bias cancels; `oracle` uses true physics. The robot never picks a goal.
Checks: with no delay the path executor reproduces the setpoint loop exactly. On 300b it pauses 5 times per minute
against 164 for the setpoint loop.

### C3 gate (frozen before any C3 numbers)

Cells: scripted operator (natural hesitations off), 1000 ticks, N = 200 paired seeds (37*1000003 + i), profiles
50j-300b. Arms `setpoint`, `chunk` (hard switch), `blend`. The safety arms `wm` and `oracle` run on 300j and 300b
with N = 100, because the CPU world model costs about 0.45 s per tick. A pause is the robot's setpoint frozen for 3
or more ticks.

C3 **PASSES** if on both 300j and 300b, `blend` has fewer pauses per minute than `setpoint` (paired-bootstrap 95% CI
excluding 0), and its coverage AUC is not worse by more than 0.01 (lower CI bound of blend - setpoint > -0.01).
Secondary, reported regardless: jerk, success, `chunk` vs `blend`, and, for the safety arms, robot-initiated block
motion (block px moved while the robot ran without a fresh command), stops per minute and their coverage cost.
Also: how well `wm` agrees with `oracle`.

## C4 + RQ4: self-auditing demonstrations

`c4.py` injects labelled faults into scripted-operator demonstrations (natural hesitations off, so clean episodes
are clean): `hesitate` (cursor frozen 20-40 ticks), `wrong` (a 15-30 tick push toward a random point), `spike`
(+700 ms on both links for 25 ticks), `slip` (the block jumps 15-30 px / 0.15-0.3 rad). Segments are 15 ticks.

Pilot (`data_c4pilot`, 180 episodes, used only to choose the score; never reused). Segment AUROC vs clean episodes:

| fault | dyn | stale | blind |
|---|---|---|---|
| hesitate | 0.39 | 0.40 | 0.45 |
| wrong | 0.73 | 0.64 | 0.66 |
| spike | 0.47 | 0.59 | 0.58 |
| slip | 0.81 | 0.68 | 0.76 |

A world model fed the executed commands cannot see faults that live in the action rather than the scene: during a
hesitation or a spike the robot holds its last command and the scene goes quiet, so prediction error drops.
It does see physics surprises (slip) and, partly, mistaken pushes. That is a finding, not a bug to tune away.

### RQ4 gate (frozen before any RQ4 data)

Data: fresh seeds. Demos: 100 episodes per delay profile (600), 1000 ticks, 30% of them with one injected fault
(equal split over the four kinds). Conformal calibration: 20 clean episodes per profile; a further 20 per profile
held out to check the false-alarm rate. Score: `dyn` (latent k-step error, K = 10, under executed commands), each
anchor credited to every tick of its window. Split conformal per delay profile (the profile is observable from the
frame stamps), alpha = 0.05; a segment above its profile's threshold is flagged.

Policy: state-based diffusion policy (2 observation ticks of agent xy, block xy, block angle sin/cos; a 16-tick chunk
of executed commands, 8 executed), trained on robot-side (state, executed command) pairs. 3 training seeds per arm.
Evaluated autonomously (no network delay) on 200 paired held-out scenes, 1000 ticks; metric coverage AUC.

Arms: `full`; `wm` (drop flagged segments) - primary; `random` (drop the same number of segments, as contiguous
15-tick blocks); `oracle` (drop the true fault windows) - diagnostic; `rules` (flag segments whose command age or
command stall exceed conformal thresholds from the same calibration set) - declared secondary.

RQ4 **PASSES** if AUC(wm) - AUC(random) > 0 with a paired-bootstrap 95% CI excluding 0, and AUC(wm) is no more than
0.01 below AUC(full) (fewer demos, no worse policy). Reported regardless: oracle and rules arms, per-fault oracle
arms, flag precision/recall and event recall per fault kind, per-profile AUROC, and the empirical false-alarm rate on
the held-out clean set next to alpha. If `oracle` does not beat `random`, these faults do not hurt this policy and
RQ4 is null for any detector.

## Running

```bash
./fleet.sh 91                          # sync + venv + self-checks
ssh yashdewangan@10.81.30.91 'cd ~/aktoria_teleop && ./smoke.sh'
```
