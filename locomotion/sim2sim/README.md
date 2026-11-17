# S4: Isaac → MuJoCo Sim2Sim（已实现，双门 PASS）

对应冻结协议 [EXPERIMENT_DESIGN.md](../../docs/EXPERIMENT_DESIGN.md) 的 **RQ4 / S4**：
在**零策略改动**的前提下，把 S1/S2 在 MuJoCo 中复刻，量化 Sim2Sim 差距
（成功率阈值 ≥ 90%），并归因主导因素。

## 结果（详见 [docs/RESULTS.md](../../docs/RESULTS.md) 与 reports/evidence/）

| 实验 | 结果 |
| --- | --- |
| 速度跟踪（S0，20 种子） | vx_rmse 0.0773 ± 0.0285（门禁 ≤0.15 ✅）；fall 0.05 |
| 四阶段序列（分层，20 种子） | 20/20（门禁 ≥0.90 ✅），平均 474 步 |
| 四阶段序列（单层，20 种子） | 0/20，fall 1.0（与 Isaac 侧失败模式一致） |
| 未见几何泛化 | 分层 80/80（含 3 m/60°/120°），单层 0/80 |
| 精度轴（容差 0.30→0.05 m） | 六档全部 20/20，终点误差 0.232→0.036 m |

## 已知的两侧语义差异（重要）

Isaac 侧阶段机在每个控制步被推进**两次**（`stage_reward` 与 `sequence_complete`
各调用一次 `_advance`，Isaac Lab 的 termination 先于 reward 计算），因此
`STOP_HOLD_STEPS = 50` 的名义 1 s 保持实际只有 **≈0.5 s**。MuJoCo 移植每控制步
只 `update()` 一次，实现的是**名义语义（1 s）**——比 Isaac 侧实际执行的**更严格**。
两侧的推进条件、容差、常数由 `tests/test_sim2sim_stage_logic.py` 的 9 项 parity
测试钉住（常数逐位一致）。完整说明见
`locomotion/sequence/stage_logic.py` 模块 docstring。

**结论口径**：S4 的 20/20 是在更严格的 stop 语义下取得的——跨引擎结论保守成立。

## 冻结约束

- 与 Isaac 侧使用**同一评估种子集** `E = {0..19}`（`locomotion/eval/seeds.py`）；
- 指标复用 `locomotion/eval/metrics.py` 的单一实现，禁止在 MuJoCo 侧另写一套口径；
- 每个被归因因素的切换单独记录为一次 run，
  按仓库证据约定生成 `run_config.json` / SHA-256 清单。

## 已完成的归因

单因素消融（详见 [docs/RESULTS.md](../../docs/RESULTS.md) 与历史报告）：
被动关节阻尼（2.0 → 0）与足部 solref 刚度是 Sim2Sim 差距的主导因素。

## 明确边界

- 两侧阶段语义的 stop-hold 差异如上——**移植更严**，非更松；
- C++ 控制栈（`cpp/`）是独立数值验证产物，**不在**本闭环内；
- 评测均在 MuJoCo 3.11；Isaac Sim 5.1.0.0 侧仅用于训练与原始评估。
