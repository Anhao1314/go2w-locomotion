# 近三年（2023–2026）文献研究：采集变量与实验流程惯例

> **用途**：为 `go2-sequence-v1` 数据集的字段设计（[DATASET.md](../DATASET.md) v1.1）与实验流程提供文献依据；同时作为论文"相关工作（近年进展）"的素材库。
> **检索日期**：2026-10-27。**方法**：4 路并行检索（arXiv API / arXiv HTML 检索页 / Semantic Scholar API / 官方文档与仓库页）。
> **证据边界**：Semantic Scholar 全程 429 限流，venue/引用状态部分未取得；多数条目为**摘要级**证据，字段级细节未核实处均已标注。⚠️ = 引用前需补全文核实。

## A. 分层 RL + 足式运动控制（2023–2026）——与 A/B 对比最相关的近期工作

| 文献 | 年份/状态 | 关键点（与我们的关系） |
| --- | --- | --- |
| [2506.20036](https://arxiv.org/abs/2506.20036) 分层 RL + 价值优化挑战地形四足 | 2025 | 两层（高层选目标 + 低层含落足 actor-critic）；**对端到端报更高回报、更少碰撞**——与我们的 A/B 同型的显式对照 |
| [2403.03848](https://arxiv.org/abs/2403.03848) 狭窄 3D 空间灵巧运动 | 2024 | 层次化控制器**优于端到端与参数化技能**，含真机 |
| [2401.03337](https://arxiv.org/abs/2401.03337) MTAC 多步态分层控制器 | 2023（ICRA'24 投稿） | HRL 多步态地形自适应，报逐任务成功率与算力开销 |
| [2403.00187](https://arxiv.org/abs/2403.00187) 受限空间 3D 表示 | 2024（ICRA'24） | 高层输出 **6D 命令** + 体素表示；程序化地形泛化 |
| [2405.01792](https://arxiv.org/abs/2405.01792) 轮足机器人导航与运动 | 2024 | 三层集成 + 特权学习；城市级部署 |
| [2411.07104](https://arxiv.org/abs/2411.07104) 多智能体长程推物 | 2024/2025 | 高/中/低三层；**成功率 +36.0%、耗时 −24.5%**（成功率+时间双指标惯例） |
| [2610.10297](https://arxiv.org/abs/2610.10297) 能耗导向 HRL 步态自适应 | 2026（ICRA'27 投稿，⚠️） | **能耗 CoT 为一级目标**；高频执行与低频自适应分离 |
| [2610.05855](https://arxiv.org/abs/2610.05855) 双足无碰分层 locomotion | 2026（⚠️） | **高层速度指令 + 底层跟踪策略**——与本项目 S1/S0 接口同构 |
| [2308.15143](https://arxiv.org/abs/2308.15143) 四足敏捷与嬉戏（生成式） | 2023（Nature MI 2024 据 comment） | 原语/环境/策略三级可预训练知识 |
| [2606.07506](https://arxiv.org/abs/2606.07506)、[2608.19977](https://arxiv.org/abs/2608.19977) | 2026 预印本（⚠️） | affordance 三级层次；跨技能动态切换（窄门跳跃） |

**要点**

1. "分层 vs 端到端"是近三年的**显式对照习惯**（2506.20036、2403.03848）——我们的受控 A/B 与文献惯例一致，可在论文中作为外部佐证（注意任务域不同，不能替本地实验背书）。
2. **接口同构**：高层速度指令+底层跟踪（2610.05855）、高层 6D 命令（2403.00187）→ 数据集应显式标注接口类型（`high_output: velocity_2d → low_output: joint_target`）。
3. **失败归因是空白**：检索范围内没有阶段级失败定位或失败模式分类的先例（最接近的是 2506.20036 的"碰撞计数"、2610.10297 的"端到端耦合难归因"的机理表述）→ 我们的"失败轨迹保留 + 阶段级归因"按**填补空白**论述，而非跟随惯例。
4. 指标趋势：**成功率 + 效率双指标**（时间/步数/能耗）。

## B. 评估与记录惯例（2023–2026）

- **指标族**：速度跟踪（带/RMSE）、稳定性与存活、摔倒、**能耗/运输代价 CoT**、OOD 性能退化、表征健康度/有效秩（[2609.06958](https://arxiv.org/abs/2609.06958)）。
- **验证阶梯**：仿真 → **sim2sim** → 硬件（[2610.20936](https://arxiv.org/abs/2610.20936)）；无真机时 sim2sim 是**最强的外部有效性证据**（[2604.11090](https://arxiv.org/abs/2604.11090) 用本体感分布匹配量化仿真差异）。
- **反 cherry-picking**：只在收敛后的策略上做迁移与报告，checkpoint 选择规则必须写清（[2504.15414](https://arxiv.org/abs/2504.15414)）——与本仓库"收敛即停 + 冻结种子"一致。
- **统一观测模板 + 动作约定 + 控制节奏**需显式声明，是跨条件可比性的前提（[2605.19503](https://arxiv.org/abs/2605.19503)）；奖励应为**可分解多分量**而非单一标量。
- 离线 RL 基准把稳定性与快速适应作为两大比较维度（[2309.16718](https://arxiv.org/abs/2309.16718)）；OOD 退化被显式评估（[2509.13737](https://arxiv.org/abs/2509.13737)）。
- ⚠️ 种子数/episode 数的具体惯例在摘要层**未取得数据**——论文中不得声称"文献常用 N 个种子"。

## C. 数据集格式与发布惯例

| 格式 | 字段命名 | 元数据/结构 | 备注 |
| --- | --- | --- | --- |
| **LeRobotDataset v3**（HuggingFace） | `observation.state`、`action`、`timestamp`、任务 ID 映射 | `meta/info.json`（features 形状/dtype/FPS/版本）+ `stats.json`（逐特征 mean/std/min/max）+ `tasks.jsonl`；Parquet 分片，**episode 边界靠元数据而非文件名** | 真机模仿学习的事实标准；写盘需 `finalize()` 后 `push_to_hub()` |
| **RLDS**（Open X-Embodiment） | Step 必填 `is_first`/`is_last`；可选 `observation`/`action`/`reward`/`is_terminal`/`discount`；episode 元数据建议键 `episode_id`/`invalid` | TFDS；`is_last` 后 action/reward 无效；`is_terminal=False` 收尾 = 截断 | 60 源数据集、1M+ 轨迹；终止/截断语义标准 |
| **HDF5-robomimic** | 复数无前缀：`actions`/`rewards`/`dones`/`states`；观测分组 `obs/<key>` | 单文件、`data/demo_i`；`env_args` JSON；**筛选用 `mask/` 不删数据**；动作归一化 −1..1 | 面向 MuJoCo/robosuite，对 Isaac Lab Go2 迁移收益低 |
| **LocoMuJoCo** | 只存**关节位置与速度**（省内存），其余加载时 FK 重算；缓存另存 | HF 数据集仓库自动下载；AMASS 因许可单独安装 | 四足/人形模仿基准；"最小存储"哲学 |
| **LAFAN1** | 逐帧通道未在页面列出；评估用 global quaternions/positions | BVH 文本；30 fps；77 段 / 496,672 帧 / 5 受试者；CC BY-NC-ND 4.0 | **主题含 "Fall and get up" 6 段**——摔倒是一等公民；卡片给各主题分布计数 |

- **数据集卡片**（HF 官方）：YAML 元数据（`license`/`tags`/`task_categories`/`size_categories` 等）+ 正文说明**潜在偏见与正确使用边界**；⚠️ 正文模板章节名未核实。
- **发布惯例**：数据放外部不可变存储（HF Hub / GCS），仓内留卡片与清单。我们的 240 条 / 40–60 MB 属小数据集，入库或 Release/Hub 均符合常规。

## D. 运动模仿管线的字段惯例（MimicKit / LocoMuJoCo）

- **MimicKit**：每帧 = `[root position(3), root rotation(3, 指数映射), joint rotations]`；**关节顺序 = 运动学树深度优先、与 XML 一致**（"顺序即定义"，必须文档化）；`.pkl` 一段一文件。
- **共同哲学**：逐帧只存**最小充分集**（姿态/关节位置速度），派生量由消费端引擎重算。
- **对我们的意义**：我们承诺"无引擎回放"（`body_pos`），比它们更自包含；但正因无引擎消费，**接触与力矩这类"有引擎时可省略"的量必须落盘**——否则不可重建。

## E. 落地修订：go2-sequence-v1 v1 → v1.1（已写入 DATASET.md）

| # | 修订 | 依据 |
| --- | --- | --- |
| 1 | 新增 `foot_contact_force [T,4] f32`（连续力；`foot_contact` bool 由阈值派生，口径文档化） | D（接触量在引擎侧被省略，我们不能省略） |
| 2 | 新增 `joint_torque [T,12] f32`（施加力矩） | 2610.10297、2509.06342：能耗是近三年一级指标，力矩是唯一不可替代输入 |
| 3 | 新增逐 step `cmd [T,3]`（vx,vy,wz；vy≡0） | 2605.19503、2610.01612：统一口径计算速度跟踪误差 |
| 4 | 新增逐 step `terminated`/`truncated` + `termination_reason`/`fail_step` | 2309.16718、2509.13737：失败需时序定位 |
| 5 | 名称/顺序表：`joint_names`/`body_names`/`foot_names` + 顺序注明（DFS、与配置一致） | MimicKit 惯例（顺序即定义） |
| 6 | manifest 增 schema 区块：`schema_version`/`features`（dtype/shape）/`fps`/栈版本/obs 分块说明/`splits`（机器可读筛选，不删数据）/逐特征统计 | LeRobot `info.json`+`stats.json`、robomimic `mask/` |
| 7 | 数据卡片增**结局分布表** + outcome → RLDS 术语映射（success/fall=terminal；timeout=truncated；`invalid`） | LAFAN1 主题计数、RLDS 语义 |
| 8 | 明确边界语义：`obs[t] --action[t]--> obs[t+1]`、最后一步有效性 | RLDS `is_last`、robomimic `dones` |
| 9 | meta 增 `interface`、`stage_enter_steps`/`stage_durations`、`robot`/`source`/`control_freq_hz`/`ref_motion: null` | A（接口同构）、2610.10297（阶段耗时）、D（消费前提） |
| 10 | 数据卡片按 HF YAML 骨架书写（可零改动推 Hub） | C |

- **保持不做**：强制多格式导出（先"可转换"即可，如后续要 IL 复用写一个 LeRobot 导出脚本）；`gait_phase` 不凭空造——无策略内相位就标 `null` 或注明"由 foot_contact 启发的派生量"。
- **成本影响**：采集实现 0.5–1 天 → ≈1 天；总计 ≈2 天 → **≈2.5 天**。

## F. 对实验思路与流程的启示

1. **A/B 报告补效率指标**：力矩落盘后即可算能耗代理（Σ|τ·q̇|·dt）与完成时间/步数，对齐 2411.07104、2610.10297 的口径（"谁更稳"+"谁更省"）。
2. **论文方法节写明 checkpoint 选择规则**（收敛即停判据 + 只评收敛后策略 + SHA 留痕）——对齐 2504.15414，反 cherry-picking。
3. **sim2sim（S4）升级为"外部有效性"章节的核心证据**：同一 checkpoint、冻结 20 种子、换物理参数组重跑、报告性能衰减——无真机条件下最有说服力的验证（2610.20936、2604.11090）。
4. **失败归因按贡献点表述**：检索无阶段级失败定位先例（A.3）——我们的 40 条失败轨迹 + `stage`/`failure/deep` 归因是差异化内容。
5. **相关工作章节素材**：A 表十篇可直接进"近年进展"段（venue 状态按 ⚠️ 标注处理）。

## G. 未核实项与后续补检

- Semantic Scholar 全局限流 → venue/卷期多数未取得；⚠️ 项引用前补全文（重点：2506.20036、2403.03848、2610.10297 的方法与实验节）。
- 未核实：LeRobot v2 精确列名；HF 卡片正文模板章节名；LocoMuJoCo 的 npz 字段名。
- 2026 年 `2610.*`/`2608.*`/`2606.*` 条目均为 arXiv 预印本，评审状态不明。