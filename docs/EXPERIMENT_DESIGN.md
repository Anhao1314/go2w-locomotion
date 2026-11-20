# Experiment Design — M1/M2 Pre-Registered Protocol

> **Status:** Pre-registered (v1.0, 2026-10-22). Evaluation seeds, metrics, and acceptance thresholds are frozen **before** any M1 training run. Any post-hoc change must be recorded in the changelog below with justification.
>
> 定位：关节级 RL locomotion 的预注册实验协议。RQ 与场景集详见重构升级方案 §5.5；本文件是执行层契约。

## 1. Research questions

- **RQ1 (feasibility & reproducibility):** Can a joint-level PPO policy for Unitree Go2 achieve stable velocity tracking in Isaac Lab, reproducible across ≥3 independent training seeds?
- **RQ2 (attribution):** Under an identical evaluation protocol, what is the **incremental** contribution of RL over (C0) a scripted PD baseline and (C1) a convex MPC baseline (`go2-convex-mpc`, MuJoCo + Pinocchio)? A negative result is reportable and acceptable.
- **RQ3 (generalization source):** How much does domain randomization contribute to generalization to unseen disturbances and unseen command combinations? (Ablation ladder: DR-off → DR-on → +teacher-student.)
- **RQ4 (transferability):** With zero policy changes, what is the Isaac → MuJoCo Sim2Sim gap, and which factors dominate (actuator model / latency / contact parameters)?

## 2. Platform & task

| Item | Value |
| --- | --- |
| Robot | Unitree Go2 (official Isaac Lab asset) |
| Task | Velocity tracking: commands vx ∈ [-1.0, 1.0] m/s, ωz ∈ [-1.0, 1.0] rad/s, vy = 0 (M1) |
| Observation | Blind, 48-dim (official `Isaac-Velocity-Flat-UnitreeGo2` spec, measured); perceptive 235-dim deferred to S3 (cloud stretch) |
| Training | PPO (RSL-RL), 2048 parallel envs, headless, Isaac Lab 2.7.0 + Isaac Sim 5.1 + Python 3.11 |
| Seeds | T1/T2/T3 = three independent training seeds (logged in each run's config); ≥3 seeds are **required for any RQ1 claim** |

## 3. Evaluation protocol (frozen)

1. **Episode seeds:** fixed evaluation seed set `E = {0..19}` (20 episodes per condition), committed in `scripts/eval_protocol.py` before training starts.
2. **Metrics (single source of truth):**
   - `vx_rmse`, `ωz_rmse`: command-tracking RMSE over the full episode, computed from base velocity after the first 0.5 s (excludes reset transient).
   - `fall_rate`: episodes terminated by base-height/contact violation.
   - `recovery_time`: seconds from disturbance offset until vx error < 0.1 m/s (S2 only).
   - `success_rate`: fraction of episodes completing without fall and with terminal position inside ±0.5 m of the commanded-integral target (S4 only).
   - Energy per meter (J/m) — reported, not gated.
3. **Reporting rule:** multi-seed results are reported as mean ± std across T1–T3. **Cherry-picking the best seed is prohibited**; a seed failing the threshold must be reported as a failure.
4. **No post-hoc metric changes.** If a metric proves ill-defined mid-experiment, the run is marked `protocol_violation` in the summary and excluded from acceptance claims (kept as negative evidence).

## 4. Acceptance thresholds (frozen; M1 smoke run may recalibrate once, then re-freeze)

| Metric | Threshold |
| --- | --- |
| Flat-ground `vx_rmse` | ≤ 0.15 m/s |
| Flat-ground `ωz_rmse` | ≤ 0.2 rad/s |
| Multi-seed consistency | All seeds meet thresholds, or the deviation is explicitly reported |
| Disturbance recovery (S2, 100 N / 0.5 s lateral push) | no-fall ≥ 90% |
| Sim2Sim success rate (S4) | ≥ 90% |
| vs C0 / C1 baselines | incremental deltas **quantified and reported**, sign unconstrained |

## 5. Scenario ladder

| Scenario | Configuration | Feeds | Timing |
| --- | --- | --- | --- |
| S1 flat velocity tracking | 2048 envs, blind 48-dim, random commands | RQ1 RQ2 | M1 |
| S2 disturbance & robustness | 100 N / 0.5 s pushes, ±3 kg payload, friction step; ablation A1–A3 (DR-off / DR-on / +distillation) | RQ2 RQ3 | M2 |
| S2b light terrain | low-amplitude rough ground (noise 2–5 cm) + slopes ≤ 15° | RQ3 | M2–M3 |
| S3 stair curriculum (cloud stretch) | official Rough composition incl. `HfPyramidStairs` 5–13 cm, perceptive 235-dim | RQ3 | budget-gated |
| S4 Sim2Sim | S1/S2 replicated in MuJoCo (same command sequence, same seeds) | RQ4 | M2–M3 |

## 5b. Execution status (updated 2026-11-16)

The protocol above is frozen; this table records **what has actually been run**.
A scenario that is not in here has not been executed, regardless of being listed above.

| Scenario axis | Status | Evidence |
| --- | --- | --- |
| S2 · 100 N × 0.5 s push | ❌ executed, gate failed (0/20 no-fall) | `reports/evidence/m2-campaign-2026-11-14/generalization_and_s2.json` |
| S2 · payload 0 / 1 / 2 / 3 kg | ✅ executed (MuJoCo, 20 seeds each): fall rate 0.05 → 0.20 / 0.20 / 0.25, vx_rmse 0.077 → 0.082 | `reports/evidence/m2-campaign-2026-11-16/payload_{0,1,2,3}kg.json` |
| S2 · friction step | ✅ executed (friction ×0.7/0.5/0.3 at t = 5 s): vx_rmse 0.077 → 0.062 / 0.064 / 0.102, fall rate 0.05 → 0.00 / 0.00 / 0.05 | `reports/evidence/m2-campaign-2026-11-16/friction_{100,07,05,03}.json` |
| S2 · DR ablation A1–A3 | ⚠️ partially: the isolation study isolates passive joint damping and foot solref, but the DR-on retraining was never run | `locomotion/sim2sim/README.md` |
| **S2b · light terrain** | 🏗️ **implemented + smoke-passed** (2026-11-20): `LIGHT_TERRAIN_CFG` filled with the protocol spec (noise 2–5 cm, slopes ≤ 15°) using Isaac Lab's BSD-3 generator; smoke on real Isaac env: sub-terrains mounted, blind confirmed, 200-step rollout stable, base heights vary across envs. **Training in progress** (seed 0, 2048 envs) | `reports/evidence/s2b-2026-11-20/` |
| S3 · stairs | ❌ not executed (budget-gated by design) | — |
| S4 · Sim2Sim | ✅ executed (velocity + sequence + unseen geometry) | `reports/evidence/s4-sim2sim-2026-11-13/` |

## 6. Baselines

- **C0 scripted PD:** standing + proportional command tracking on the same task definition (lower bound).
- **C1 convex MPC:** `elijah-waichong-chan/go2-convex-mpc` integrated into the same evaluation loop in MuJoCo; license reviewed before adoption; any wrapper code is registered in `THIRD_PARTY_NOTICES.md`.
- Baselines run on the **same episode seeds and metrics** as the RL policies. Pairing by seed is required for RQ2 claims.

## 7. Artifacts & provenance

- Every run directory: `run_config.json` (git SHA, dependency versions, seeds, command line), `events.jsonl`, `eval/` summaries, and a SHA-256 manifest — consistent with the existing evidence-bundle convention.
- Checkpoints are released per training-seed tag together with `VecNormalize` statistics when applicable.
- Negative or inconclusive results are archived, not deleted.

## Changelog

| Date | Change |
| --- | --- |
| 2026-10-22 | v1.0 pre-registered before any M1 training run |
| 2026-10-22 | Version correction (before any training run; no protocol-semantics change): §2 table said "Isaac Lab 2.3.2", but the actual environment prepared for M1 is **Isaac Lab 2.7.0** (isaaclab 0.54.2, editable install of `D:\work\IsaacLab`) + **Isaac Sim 5.1.0.0** (isaacsim 5.1.0.0) + Python 3.11 (repo `.venv311`). Recorded per the freeze rule so the environment of record matches what will actually execute M1; Isaac Sim version was already correct. |
| 2026-10-22 | Observation-dimension correction (before any training run): the protocol said "blind 87-dim", but the interface of record was measured on the actual task (`Go2Loco-S1-Flat-Blind-v0`, Kit 5.1 + Isaac Lab 2.7.0): policy obs is **48-dim** = base_lin_vel 3 + base_ang_vel 3 + projected_gravity 3 + velocity_commands 3 + joint_pos 12 + joint_vel 12 + actions 12. The perceptive 235-dim figure is consistent (48 + 187 height scan) and unchanged. 87 is treated as a transcription error; thresholds and metrics are unaffected. |
