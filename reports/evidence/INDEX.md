# 证据索引（`reports/evidence/`）

本目录存放**评测输出的原样副本**，是 [`docs/RESULTS.md`](../../docs/RESULTS.md) 里每个数字的凭据。
工作目录 `logs/` 被 `.gitignore` 排除，因此凡是要进入结论的证据都会先落一份到这里。

原则：

1. **不改数值**。只做「复制」，不重算、不裁剪字段（文件体积小，保留全量 episode 明细）。
2. **保留可追溯性**。每份证据记录生成时间、checkpoint、协议标签、种子集合。
3. **负结果同样入库**。未过门槛的结果不删、不隐藏。

---

## s4-sim2sim-2026-11-13/

来源：`logs/s4/*_20261113_*/evaluation.json`（MuJoCo 3.11，冻结种子 `E={0..19}`）

| 文件 | 实验 | 关键结论 |
|---|---|---|
| `velocity_s0.json` | S4 速度跟踪（S0 策略，`Go2Loco-S1-Flat-Blind-v0`） | vx_rmse 0.0773 ± 0.0285（门禁 ≤0.15）；wz_rmse 0.1391；fall 0.05 |
| `sequence_layered.json` | S4 四阶段长程序列（分层） | 20/20 成功；平均 474 步；fall 0 |
| `sequence_single.json` | S4 四阶段长程序列（单层） | 0/20；fall 1.0（与 Isaac 侧失败模式一致） |

协议备注：单层与分层使用**同一观测、同一任务、同一冻结种子**，唯一变量是有无显式分层。
评测耗时 20 段序列约 4 秒（MuJoCo，无渲染），可低成本重跑。

---

## m2-campaign-2026-11-14/

来源：`logs/s4/s2_generalization/evaluation.json`、`logs/s4/c0_baseline/evaluation.json`

| 文件 | 实验 | 关键结论 |
|---|---|---|
| `generalization_and_s2.json` | ① S2 扰动鲁棒性 ② 未见几何泛化（分层/单层） | ① **0/20 无摔倒**（门禁 ≥0.90，未过）；侧推 100 N × 0.5 s，第 5 秒施加 ② 分层 **80/80 = 1.00**（4 几何 × 20 种子，含 3 个未见几何），单层 **0/80** |
| `c0_baseline.json` | RQ2 下界：脚本小跑（PD 正弦，无学习） | 无推力 **0.6064** / fall 0；同一 100 N 推力 **0.5759** / no-fall **0/20**（recovery 0.48 s）→ RL 相对脚本降低误差 **7.8×**，但两者在同一推力档位下都摔倒 |

协议备注：

- S2 的 100 N 侧推是**预注册协议**里的档位；训练时未启用推力域随机化，因此 0/20 是
  **协议有效性的证据**（训练分布缺失被暴露），不是实现事故。完整根因与单因素归因见
  `locomotion/sim2sim/README.md`。
- 泛化矩阵的「未见几何」指训练阶段阶段参数只在 `2 m / 90°` 上采样；`3 m / 90°`、
  `2 m / 60°`、`2 m / 120°` 均为评测时首次出现。

---

## ros2-r1-2026-11-13/

来源：`logs/s4/ros_live/attempt_007/probe_receipt.json`（真 DDS，两独立进程）

| 文件 | 实验 | 关键结论 |
|---|---|---|
| `probe_receipt.json` | ROS-R1 级闭环验收：`/go2/cmd_vel` 进，`/clock` `/joint_states` `/base_pose` `/tf` `/session_info` 出 | `ROS2_ROUNDTRIP_PASS`；四话题各 582 条；TF 校验通过；前进 1.2384 m；转向 +1.5853 rad；接受 187 条指令 |

协议备注：attempt_001–006 为负证据，全部留档在 `ros2/go2_bridge/acceptance.md`：001 容器不可用
（Docker daemon 未启动，改走 WSL2 路线）、002 pip 被 PEP 668 拦截、003 probe 缺 `rclpy.init()`
且依赖缺 `gymnasium`、004 TF 已发布但探针未订阅校验、005/006 probe 内 `TFMessage` 作用域错误。
CI 中的 `ros2-live-r1` 工作流会重跑同类验收并硬门禁。

---

## vla-2026-11-15/

来源：`logs/vla/openloop_check_060k/openloop_summary.json`

| 文件 | 实验 | 关键结论 |
|---|---|---|
| `openloop_060k.json` | 开环复现检查（LIBERO SmolVLA，60K 步 checkpoint） | 40 帧上动作 MSE **0.0656** vs 平均动作基线 0.2117（比值 **0.31**）；七维相关 +0.82/+0.91/+0.78/+0.64/+0.58/+0.58/+0.85；夹爪符号一致 **0.93**；遮住图像后动作偏移 L2 **0.489** |
| `libero_seeded_a.json` / `libero_seeded_b.json` | 闭环评测，种子固定（1000–1009 / 1010–1019），fps=20 | **批 A 7/10、批 B 3/10 → 合并 10/20 = 0.50**（权威数字） |
| `libero_ab_fps20_n10.json` / `libero_ab_fps10_n10.json` | 控制频率 A/B（未固定策略种子） | fps=20 7/10；fps=10 **0/10**（140 步上限全部超时） |
| `libero_video_pack.json` | 视频包录制（seed 1000/1003/1006/1001） | 2 成功 + 2 超时，用于 `media/training/libero_quad.mp4` 与 `libero_dual.mp4` |

协议备注：

- 采样方式：8 个 episode 均匀分布，每段取 10%/30%/50%/70%/90% 五帧，共 40 帧。
- 输入走的是 checkpoint 自带的预处理管线（含训练时保存的归一化统计量），与训练/评测一致。
- 该测试**只检验「策略能否复现训练演示」**，不检验闭环成功率；后者仍未测（见 RESULTS.md）。
- 附带发现：训练数据 10 Hz，而 LeRobot `LiberoEnvConfig` 默认 `fps=20`。A/B 结果：fps=20
  有效、fps=10 全败 —— 数据集元数据里的 fps 不是控制契约。
- **测量波动（重要）**：策略动作由 flow matching 采样，逐种子不保证跨运行一致
  （seed 1003：70 步成功 vs 280 步超时）。因此单批 10 集不可外推，闭环数字以 20 集合并为准。

---

## 未入库但存在的证据（工作目录）

| 路径 | 内容 | 为何未入库 |
|---|---|---|
| `logs/rsl_rl/**`、`logs/rsl_rl_layered/**` | Isaac Lab 训练 checkpoints 与 `run_config.json` / SHA-256 清单 | 体积（约 1.9 GB） |
| `logs/vla/smolvla_run1/**` | 教师数据微调 checkpoint（20K，13 GB） | 体积 |
| `logs/vla/libero_smolvla/**` | LIBERO 微调 checkpoint 10K–60K（7.4 GB） | 体积 |
| `logs/vla/eval_run6.log` | 教师策略跨域评测：1 个 episode、3 600 步、放置 0/4 方块 | 单条日志，已在 RESULTS.md 中引用 |
| `logs/vla/train_libero_00*.log` | LIBERO 训练日志（loss 曲线原始记录） | 工作日志 |
