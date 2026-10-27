# 轨迹数据集规格（v1.1 修订定稿）

> **版本**：v1 定稿 2026-10-27；**v1.1 修订 2026-10-27 晚**——依据近三年（2023–2026）文献研究补充字段与惯例（依据见 [notes/recent-papers-2023-2026-data-practices.md](notes/recent-papers-2023-2026-data-practices.md)，逐条对应其 §E）。
> **定位**：回应"数据集/采集器"关切——本课题走**自建数据集**路线：仿真轨迹正式导出为交付数据集（论文贡献之一），同时兼作评估原始证据与后续复用素材。
> 与 [EXPERIMENT_DATA.md](EXPERIMENT_DATA.md) 的关系：那份协议管"**每 run 证据束**"（配置/事件/哈希，单次运行）；本规格管"**跨 run 的轨迹数据集**"（逐 step 状态-动作序列，按条件成集）。互补，不重复。

**数据集名**：`go2-sequence-v1`（Go2 长程运动任务轨迹数据集）
**落盘位置**：`data/datasets/go2-sequence-v1/`

## 修订记录

| 版本 | 日期 | 变更 |
| --- | --- | --- |
| v1 | 2026-10-27 | 定稿：采集范围 240 条、逐 step 字段、存储与校验、采集命令与成本 |
| v1.1 | 2026-10-27 晚 | 依文献研究补：`joint_torque`、`foot_contact_force`（连续力）、逐 step `cmd`、逐 step `terminated/truncated`、名称与顺序表、manifest schema/统计/splits、RLDS 边界语义、结局分布表、接口标注；成本 2 → 2.5 天 |

---

## 1. 采集范围（240 条 episode）

| 来源 | 条件 | 条数 | 说明 |
| --- | --- | --- | --- |
| **L3 主实验**（核心） | 分层 3 训练种子 × 20 评估种子 | 60 | 全部成功轨迹 |
| | 单层 3 训练种子 × 20 评估种子 | 60 | **含 40 条失败轨迹**（归因证据，必须保留） |
| L1 对比 | 分层/单层各 1 种子 × 20 | 40 | 简单任务参照 |
| 泛化 | 分层 4 种任务几何 × 20 | 80 | 含 3 个未见几何 |
| 合计 | | **240** | 单层泛化待补测（超时问题排查后 +≤80，作为 v1.1 数据追加） |

**不采集**：S0 速度跟踪轨迹（v1 排除，避免膨胀；论文如需另开 v2）；相机图像（本环境 `enable_cameras` 崩溃且体积大）；训练过程 rollout（海量低价值）。

## 2. 每条 episode 字段（控制频率 50 Hz，逐 step）

| 字段 | 形状 | 来源 / 说明 |
| --- | --- | --- |
| `obs` | [T, 53] f32 | 策略观测原样；**分块构成须在 manifest 与数据卡片中逐段列出**（实现时从 observation_manager 术语顺序生成，不手工誊写） |
| `action` | [T, 2 或 12] f32 | 策略原始动作：分层 = 速度指令（vx, wz），单层 = 12 维关节位置目标 |
| `cmd` | [T, 3] f32 | **逐 step 任务级指令**（vx, vy, wz；vy≡0）：分层组即策略动作；单层组为闭环阶段指令——用于跨组同口径计算速度跟踪误差 【v1.1】 |
| `joint_target` | [T, 12] f32 | 实际下发关节目标：单层取 action_manager 处理结果；分层取 wrapper 内 S0 输出 |
| `joint_torque` | [T, 12] f32 | **施加力矩**（Isaac Lab `applied_torque`）——能耗/CoT 分析唯一不可替代输入 【v1.1】 |
| `base_pos` / `base_quat` | [T,3] / [T,4] f32 | 世界系位置 / 姿态（wxyz） |
| `base_lin_vel_b` / `base_ang_vel_b` | [T,3] ×2 f32 | 机体坐标系线/角速度 |
| `joint_pos` / `joint_vel` | [T,12] ×2 f32 | 关节位置/速度 |
| `foot_contact` | [T, 4] bool | 足端触地（由 `foot_contact_force` 按阈值派生，阈值与奖励口径一致并文档化）【v1.1 改为派生】 |
| `foot_contact_force` | [T, 4] f32 | **连续接触力**（官方 `contact_forces` 传感器，`.*FOOT`）——bool 在阈值附近丢失信息且无引擎不可重建 【v1.1】 |
| `stage` | [T] int8 | 任务阶段标签 0–3（L1 为 0–1）；语义在数据卡片中写明 |
| `reward` | [T] f32 | 逐步**总**奖励；奖励分项名与权重记入 meta（逐 step 分项列为可选，实现成本低则追加）【v1.1】 |
| `terminated` / `truncated` | [T] ×2 bool | 逐 step 终止/截断标记（失败时序定位用；RLDS 语义）【v1.1】 |
| `body_pos` | [T, 13, 3] f32 | 全部连杆世界位置（可视化/无引擎回放，现有视频管线直接消费） |
| `sim_time` | [T] f32 | = step × dt（dt = 0.02 s） |
| `meta` | JSON 字符串 | 自包含元数据（见下），与 manifest 同源 |

**每条 episode 元数据**（`manifest.json` + npz 内 `meta`）：

- `episode_id`（如 `l3_layered_seed0_eval07`）、`task`、`level`（L1/L3）、`group`（layered/single）
- `training_seed`；`checkpoint` 路径 + SHA-256
- `eval_seed`（冻结种子集 E={0..19} 成员）
- `config`：**运行时实际**任务常量（前进距离、转角、容差——泛化配置的 patch 值一并记录）
- `outcome`（success / fall / timeout）、**`success` bool、`termination_reason`、`fail_step`**（首次终止步索引）【v1.1】
- **`stage_enter_steps` / `stage_durations`**：各阶段进入步与耗时——阶段边界统计一键出 【v1.1】
- **`interface`**：`{high_output: "velocity_2d", low_output: "joint_target"}`（分层）；单层组标注为 `end_to_end_joint`——与文献接口同构度显式化 【v1.1】
- **名称/顺序表**：`joint_names`、`body_names`（13）、`foot_names`；顺序注明"与配置文件一致"（顺序即定义）【v1.1】
- **`robot: "go2"`、`source: "isaaclab_sim"`、`control_freq_hz: 50`、`ref_motion: null`**（无参考动作显式写 null，便于未来模仿学习复用区分）【v1.1】
- `git_commit`、`recorded_at`、`dt`

## 3. 存储与校验

```text
data/datasets/go2-sequence-v1/
├── README.md            # 数据卡片（含 HF YAML 骨架头、结局分布表、使用边界）
├── manifest.json        # schema 区块 + 全部 episode 元数据 + 文件 SHA-256
├── checksums.sha256
└── episodes/
    └── l3_layered_seed0_eval07.npz     # np.savez_compressed，一 episode 一文件
```

- **manifest 顶层 schema 区块**【v1.1】：`schema_version`、`features`（逐字段 name/dtype/shape）、`fps: 50`、`robot`、仿真栈版本（Isaac Lab / Isaac Sim）、`total_episodes`/`total_steps`、**`obs_layout`（53 维分块说明）**、**`splits`**（机器可读的 episode_id 清单：all / success / failure / layered / single / generalization——**筛而不删**，对齐 robomimic filter key 惯例）、**逐特征统计**（mean/std/min/max，注明基于全量）。
- **边界语义（写入数据卡片）**【v1.1】：`obs[t] --action[t]--> obs[t+1]`；`reward[t]` 由 `action[t]` 触发；最后一步的 action/reward 有效性说明；`success`/`fall` = terminal，`timeout` = truncated（RLDS 术语映射）。
- **体积预估**：单条 0.3–0.5 MB（未压缩，T≈400–900；新增字段后约 +25%）；全量压缩后约 **50–75 MB**，与现有 `data/demo_trajectories/`（26 MB）同量级，**全量入库**；若超过 80 MB，改为 GitHub Release 资产 + manifest 入库（降级顺序：`body_pos` 转 f16 → 省略 `body_pos`，可由 `base_*`+`joint_*` 重算）。
- **命名**：`{level}_{group}_seed{训练种子}_eval{评估种子:02d}.npz`。
- **不强制多格式导出**：npz+manifest 为唯一主格式；后续如需模仿学习复用，再写一个到 LeRobotDataset（v3）的转换脚本（"可转换"优先于"多格式"）。

## 4. 采集方式

- **实现**：扩展 `scripts/eval_policy.py`，新增 `--record-dataset <dir> --dataset-group <label>`；把现有"仅 env0 的 body_pos 记录"扩展为全部 20 个并行 env 的逐 step 记录，episode 终止时按 env 落盘。
- **命令**（每条件一次评估重跑，复用现有冻结评估）：
  `python scripts/eval_policy.py --task Go2Loco-Sequence-v0 --checkpoint <ckpt> [--layered --s0-checkpoint <s0>] --episodes 20 --record-dataset data/datasets/go2-sequence-v1 --dataset-group l3_layered_seed0`
- **成本**：实现 ≈1 天（v1.1 字段增加后） + 采集 ≈0.5 天（12 次评估重跑；单层泛化需先排查超时）+ 卡片/校验/统计 0.5 天 ≈ **2.5 天**。
- **依赖**：单层泛化超时问题（PLAN 待办）若属脚本挂起，先修复再纳入采集。

## 5. 验收标准

1. 每条 episode 的 `outcome` / `final_stage` 与对应报告行一致（L3/L1/泛化三份报告）；
2. 每个条件的 20 个评估种子齐全，**失败轨迹不做过滤**（归因证据是交付内容的一部分）；
3. `checksums.sha256` 校验通过；manifest 声明字段全部存在且形状正确（校验脚本一并提交）；
4. 数据卡片中注明仿真限定、生成命令与**使用边界**，可一键复采。

## 6. 用途映射

| 用途 | 消费方 |
| --- | --- |
| 失败归因（阶段序列定位、终止时序） | 论文第 5 章、l3_ab_comparison.md 复核 |
| 能耗/效率指标（力矩 → Σ\|τ·q̇\|·dt） | 论文效率对比小节（对齐近三年能耗口径） |
| 运动视频（无引擎回放） | `render_motion_video.py`（`body_pos` 直接消费） |
| 模仿学习复用（obs→action 对） | 展望章；Phase-1 BC/DAgger 管线已证明此用法可行 |
| 答辩展示 | 数据卡片 + 6 条样例（每条件 1 条）现场演示 |