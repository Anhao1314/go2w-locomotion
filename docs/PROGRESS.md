# 进度看板

> **最后更新**：2026-10-24
> **当前阶段**：毕设收尾（实验主线已完成，剩补种子 / L2 / 论文）
> **单一入口**：状态以此页为准；计划见 [PLAN.md](PLAN.md)，研究问题见 [RESEARCH.md](RESEARCH.md)。

## 研究主线状态

| 研究问题 | 状态 | 结论 |
| --- | --- | --- |
| RQ1 样本效率 | ✅ | L1：分层快 3 倍（50 vs 150 迭代）；L3：分层 100 迭代收敛，单层 1500 不收敛 |
| RQ2 成功率 | ✅ | L1 双方 100%；**L3：分层 20/20 vs 单层 0/20（成败分离）** |
| RQ3 泛化 | ⏸ 未做 | 可选加分项（O1） |
| RQ4 失败可归因 | ✅ 方法已验证 | 单层 L3 失败全部定位到阶段 1（行走→转向切换）；分层无需学转向步态 |

## 已完成里程碑

| 里程碑 | 日期 | 证据卡 |
| --- | --- | --- |
| M0 环境 / 协议 / 证据束 / 面板 | 10/23 | `eval_protocol --verify` 6/6；自动报告与证据束全链路 |
| M1 S0 三种子基线 | 10/23 | `vx 0.0569±0.0011`、零摔倒；[报告](../reports/locomotion/m1_evaluation.md) |
| M2 任务环境 + 难度阶梯（L1/L3） | 10/23–24 | `locomotion/sequence/`；L3 经三次关键修复（heading 锚定 / 闭环指令 / 漂移停止） |
| M3 L1 受控对比 | 10/23 | 双方 100%；分层收敛快 3 倍；[报告](../reports/training/ab_comparison.md) |
| **M4 L3 主结果** | **10/24** | **B 20/20（100 迭代，173 s）vs A 0/20（1500 迭代，全摔）**；[报告](../reports/training/l3_ab_comparison.md)；[成功视频](../media/training/robot_motion_l3_layered.mp4) / [失败视频](../media/training/robot_motion_l3_single_fail.mp4) |
| M5 仓库毕设化 | 10/24 | README 首屏重写（中英）、Phase-1 归档、318 链接校验（`d4d1986`） |

## 剩余任务（详见 [PLAN.md §3](PLAN.md#3-剩余必做答辩前)）

| ID | 任务 | 状态 | 预算 |
| --- | --- | --- | --- |
| T1 | 补训练种子（L3，A/B 各 +2） | ⏳ | ~1.5 h |
| T2 | L2 中间难度 | ⏳ | ~1.5 h |
| T3 | README 边界更新（种子补齐后） | ⏳ | 10 min |
| T4 | 论文初稿 | ⏳ | 2–3 周 |

## 阻塞与风险

| 项 | 状态 |
| --- | --- |
| 无阻塞 | T2 若两组都失败，按计划转为负结果报告，不追加超过 2 轮重试 |

## 产物索引

| 类型 | 位置 |
| --- | --- |
| 对比报告 | `reports/training/l3_ab_comparison.md`、`ab_comparison.md` |
| S0 基线报告 | `reports/locomotion/m1_evaluation.md` |
| 全部训练报告（自动生成，11 份） | `reports/training/<run-id>.md` |
| 关键视频 | `media/training/`（L3 成功/失败、L1 成功、M1 曲线、效率图） |
| 每日日志（含失败与修正） | `docs/journal/` |
