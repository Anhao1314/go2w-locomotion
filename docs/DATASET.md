# 轨迹数据集规格 v1（定稿）

> **2026-10-27 定稿**。回应"数据集/采集器"关切：本课题走**自建数据集**路线——仿真轨迹正式导出为交付数据集（论文贡献之一），同时兼作评估原始证据与后续复用素材。
> 与 [EXPERIMENT_DATA.md](EXPERIMENT_DATA.md) 的关系：那份协议管"**每 run 证据束**"（配置/事件/哈希，单次运行）；本规格管"**跨 run 的轨迹数据集**"（逐 step 状态-动作序列，按条件成集）。互补，不重复。

**数据集名**：`go2-sequence-v1`（Go2 长程运动任务轨迹数据集）
**落盘位置**：`data/datasets/go2-sequence-v1/`

---

## 1. 采集范围（240 条 episode）

| 来源 | 条件 | 条数 | 说明 |
| --- | --- | --- | --- |
| **L3 主实验**（核心） | 分层 3 训练种子 × 20 评估种子 | 60 | 全部成功轨迹 |
| | 单层 3 训练种子 × 20 评估种子 | 60 | **含 40 条失败轨迹**（归因证据，必须保留） |
| L1 对比 | 分层/单层各 1 种子 × 20 | 40 | 简单任务参照 |
| 泛化 | 分层 4 种任务几何 × 20 | 80 | 含 3 个未见几何 |
| 合计 | | **240** | 单层泛化待补测（超时问题排查后 +≤80，作为 v1.1 追加） |

**不采集**：S0 速度跟踪轨迹（v1 排除，避免膨胀；论文如需另开 v2）；相机图像（本环境 `enable_cameras` 崩溃且体积大）；训练过程 rollout（海量低价值）。

## 2. 每条 episode 字段（控制频率 50 Hz，逐 step）

| 字段 | 形状 | 来源 / 说明 |
| --- | --- | --- |
| `obs` | [T, 53] f32 | 策略观测原样（使数据集可直接用于模仿学习复用） |
| `action` | [T, 2 或 12] f32 | 策略原始动作：分层 = 速度指令，单层 = 关节位置目标 |
| `joint_target` | [T, 12] f32 | 实际下发关节目标：单层取 action_manager 处理结果；分层取 wrapper 内 S0 输出 |
| `base_pos` / `base_quat` | [T,3] / [T,4] f32 | 世界系位置 / 姿态（wxyz） |
| `base_lin_vel_b` / `base_ang_vel_b` | [T,3] ×2 f32 | 机体坐标系线/角速度 |
| `joint_pos` / `joint_vel` | [T,12] ×2 f32 | 关节位置/速度 |
| `foot_contact` | [T, 4] bool | 足端触地，来自官方 `contact_forces` 传感器（`.*FOOT`，阈值与奖励口径一致） |
| `stage` | [T] int8 | 任务阶段标签 0–3（L1 为 0–1） |
| `reward` | [T] f32 | 逐步总奖励 |
| `body_pos` | [T, 13, 3] f32 | 全部连杆世界位置（可视化/无引擎回放，现有视频管线直接消费） |
| `sim_time` | [T] f32 | = step × dt（dt = 0.02 s） |
| `meta` | JSON 字符串 | 自包含元数据（见下），与 manifest 同源 |

**每条 episode 元数据**（`manifest.json` + npz 内 `meta`）：

- `episode_id`（如 `l3_layered_seed0_eval07`）、`task`、`level`（L1/L3）、`group`（layered/single）
- `training_seed`；`checkpoint` 路径 + SHA-256
- `eval_seed`（冻结种子集 E={0..19} 成员）
- `config`：**运行时实际**任务常量（前进距离、转角、容差——泛化配置的 patch 值一并记录）
- `outcome`（success / fall / timeout）、`final_stage`、`steps`、`fell`
- `git_commit`、`recorded_at`、`dt`

## 3. 存储与校验

```text
data/datasets/go2-sequence-v1/
├── README.md            # 数据卡片：规模、字段、生成命令、用途、边界
├── manifest.json        # 全部 episode 元数据 + 文件 SHA-256
├── checksums.sha256
└── episodes/
    └── l3_layered_seed0_eval07.npz     # np.savez_compressed，一 episode 一文件
```

- **体积预估**：单条 0.3–0.5 MB（未压缩，T≈400–900）；全量压缩后约 **40–60 MB**，与现有 `data/demo_trajectories/`（26 MB）同量级，**全量入库**；若超过 80 MB，改为 GitHub Release 资产 + manifest 入库（降级顺序：`body_pos` 转 f16 → 省略 `body_pos`，可由 `base_*`+`joint_*` 重算）。
- **命名**：`{level}_{group}_seed{训练种子}_eval{评估种子:02d}.npz`。

## 4. 采集方式

- **实现**：扩展 `scripts/eval_policy.py`，新增 `--record-dataset <dir> --dataset-group <label>`；把现有"仅 env0 的 body_pos 记录"扩展为全部 20 个并行 env 的逐 step 记录，episode 终止时按 env 落盘。
- **命令**（每条件一次评估重跑，复用现有冻结评估）：
  `python scripts/eval_policy.py --task Go2Loco-Sequence-v0 --checkpoint <ckpt> [--layered --s0-checkpoint <s0>] --episodes 20 --record-dataset data/datasets/go2-sequence-v1 --dataset-group l3_layered_seed0`
- **成本**：实现 0.5–1 天 + 采集 ≈0.5 天（12 次评估重跑；单层泛化需先排查超时）+ 卡片与校验 0.5 天 ≈ **2 天**。
- **依赖**：单层泛化超时问题（#7 待办）若属脚本挂起，先修复再纳入采集。

## 5. 验收标准

1. 每条 episode 的 `outcome` / `final_stage` 与对应报告行一致（L3/L1/泛化三份报告）；
2. 每个条件的 20 个评估种子齐全，**失败轨迹不做过滤**（归因证据是交付内容的一部分）；
3. `checksums.sha256` 校验通过；manifest 声明字段全部存在且形状正确（校验脚本一并提交）；
4. 数据卡片中注明仿真限定与生成命令，可一键复采。

## 6. 用途映射

| 用途 | 消费方 |
| --- | --- |
| 失败归因（阶段序列定位） | 论文第 5 章、l3_ab_comparison.md 复核 |
| 运动视频（无引擎回放） | `render_motion_video.py`（`body_pos` 直接消费） |
| 模仿学习复用（obs→action 对） | 展望章；Phase-1 BC/DAgger 管线已证明此用法可行 |
| 答辩展示 | 数据卡片 + 6 条样例（每条件 1 条）现场演示 |