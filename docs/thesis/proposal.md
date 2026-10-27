# 毕业设计开题报告

> 使用说明：本文为内容底稿，交前需按学校模板套版（封面、字数、格式）。
> 标注 `[待补：...]` 处为需要按学校要求/核实后填写的内容。

---

**课题名称**：基于分层强化学习的四足机器人运动控制系统设计与仿真验证
**学生**：[待补]
**专业**：[待补]
**指导教师**：[待补]

---

## 一、选题背景与意义

### 1.1 选题背景

四足机器人是具身智能落地最具代表性的载体之一。随着宇树 Go2 等平台的普及，基于深度强化学习（Reinforcement Learning, RL）的运动控制已成为该领域的主流方法：策略在 GPU 大规模并行仿真中训练，经域随机化迁移至真机，训练周期从数月缩短到小时级[1][2][3]。

然而，当前多数工作将"任务理解、决策与运动控制"全部交给一个端到端策略：网络直接从传感器观测映射到全部关节指令。这种范式在单一技能（如平地行走）上表现良好，但在**长程任务**——需要按顺序完成行走、转向、停靠等多个阶段——上存在两个已知困难：其一，任务奖励稀疏且跨阶段异质，探索难以命中有效行为序列；其二，策略是黑箱，训练失败时无法判断问题出在"决策"还是"执行"。

分层架构是应对上述困难的自然思路：由高层策略输出语义化的运动指令（如前进速度、转向速率），底层策略（或传统控制器）负责将指令落实为关节运动。该思路在工程界已被广泛采用，也得到分层强化学习理论的支持[6][7][8]。

### 1.2 选题意义

尽管分层实践普遍，**公开文献中缺少严格的受控对比**：现有分层四足工作大多只报告自身架构的成功率，不提供同等条件（同任务、同观测、同训练预算）下的端到端单层基线，导致"分层到底带来什么、代价是什么"缺乏量化答案[9]。

本课题的意义在于：

1. **工程意义**：设计并实现一套完整的四足机器人分层运动控制系统（仿真），覆盖任务定义、策略训练、评估协议与监控分析全链路，为后续真机部署提供系统基础；
2. **方法意义**：通过受控对比实验，量化分层架构相对端到端单层在训练成功率、样本效率与失败可归因性上的差异，为同类系统的架构选型提供数据支撑；
3. **教学意义**：完整实践"预注册实验协议—证据链管理—多种子验证"的科研规范，训练严谨的工程研究方法。

## 二、国内外研究现状

### 2.1 大规模并行强化学习运动控制

Hwangbo 等[1]在 Science Robotics 发表的工作将 RL 与学习执行器模型结合，使 ANYmal 四足机器人获得敏捷运动技能，确立了"仿真训练 + 真机迁移"范式；Lee 等[2]进一步提出特权学习（teacher-student）框架应对复杂地形。Rudin 等[3]提出大规模并行训练方法，在单 GPU 上以数千并行环境将四足运动策略的训练时间压缩到 20 分钟以内，并给出关键工程结论：并行规模增大时 rollout 步长应保持较小、mini-batch 应增大。Walk These Ways[4]提出行为参数化条件策略，单一策略通过调节行为参数获得多种步态，为高层指令接口设计提供了新思路。在工具层面，Isaac Gym/Isaac Lab[5]与 legged_gym 等开源框架使上述方法成为可复现的工程标准。

### 2.2 分层强化学习

分层 RL 的理论起点是 Sutton 等的 Options 框架[6]，将时间抽象动作形式化为半马尔可夫决策过程；Option-Critic[7]实现了 option 的端到端梯度学习；FeUdal Networks[8]确立了"经理输出目标、工人执行动作"的现代分层形态。Pateria 等[9]对该领域做了系统综述。在四足机器人上，Jain 等[10]提出高层策略在潜空间发指令、低层策略执行的架构，是分层四足运动的代表工作；近期工作进一步将分层与中枢模式发生器（CPG）等生物启发方法结合[11]。

### 2.3 研究缺口

综合来看：(1) 大规模并行训练已解决"单个技能学得快"的问题；(2) 分层架构有成熟理论与四足先例。但**现有分层四足工作均未提供同观测、同预算、同评估协议的端到端单层对照**——分层的收益（成功率、样本效率）与代价（接口损失、实现复杂度）因此缺乏量化证据；同时，端到端策略训练失败时缺少系统化的失败定位手段。这两个缺口构成本课题的切入点。

### 2.4 相邻方向

奖励设计方面，AMP[12]用对抗运动先验替代手工奖励，Eureka/DrEureka[13][14]用大语言模型自动生成奖励与域随机化参数；安全方面，控制屏障函数（CBF）在四足机器人上实现多层安全约束[15][16]。这些工作与本课题的"层间指令设计""可靠性分析"相互参照。

## 三、研究内容与拟解决的关键问题

### 3.1 研究内容

本课题设计并实现一套基于分层强化学习的四足机器人运动控制系统（Unitree Go2，Isaac Sim 仿真），核心研究内容包括：

1. **系统设计**：分层运动控制架构——上层（S1）指令策略在 2 维指令空间（前进速度、转向速率）学习任务级行为；底层（S0）locomotion 策略接收指令并输出 12 维关节控制；任务/阶段管理器生成结构化的闭环指令与阶段判定；
2. **对照系统**：端到端单层配置——与分层系统**完全相同的观测与任务信息**，直接输出 12 维关节动作；
3. **受控实验**：在难度阶梯（L1：直线行走+停靠；L3：行走—转向—行走—停靠四阶段序列）上，以冻结评估种子集、统一指标对两种架构进行对比，并进行阶段级失败归因分析。

### 3.2 拟解决的关键问题

1. **动作接口设计**：上层指令空间的语义选择（速度指令 vs 行为参数），以及底层观测在两级间的正确传递与对齐；
2. **长程任务的奖励设计**：跨阶段异质奖励下如何提供有效的探索梯度（本课题采用势函数差分整形 + 阶段速度引导的方案）；
3. **评估公平性**：预注册冻结种子、统一指标与停止判据，多种子验证结论稳健性。

## 四、研究方法与技术路线

### 4.1 研究方法

- **仿真平台**：Isaac Sim 5.1 + Isaac Lab 2.7，Unitree Go2 官方资产，2048 并行环境；
- **训练算法**：PPO（RSL-RL 实现），采用自适应学习率与 GAE；
- **实验设计**：预注册协议——评估种子集、指标（成功率、阶段到达率、完成步数）与判据在训练前冻结；每组架构训练多种子；训练中以确定性策略在训练环境上周期评估，连续达标自动早停并记录收敛点；
- **证据管理**：每次训练自动生成运行配置（代码版本、依赖、命令行）与全文件 SHA-256 清单；分析基于冻结数据，负结果一并归档。

### 4.2 技术路线

```text
阶段一  底层能力        Go2 速度跟踪策略（S0）训练与冻结种子评估
        ↓
阶段二  任务构建        四阶段命令序列环境（难度阶梯 L1/L3）+ 评估协议冻结
        ↓
阶段三  受控对比        单层配置 A  ─┐
                        分层配置 B  ─┤→ 同种子训练 → 冻结种子评估 → 跨种子验证
        ↓
阶段四  分析与交付      阶段级失败归因、样本效率曲线、论文撰写
```

### 4.3 可行性说明

课题依托的前期探索已完成主要技术验证：底层速度跟踪策略三训练种子均达标（速度跟踪误差 0.057±0.001 m/s，阈值 0.15）；四阶段任务环境与自动评估链路已运行；初步对比实验显示两种架构在同一任务上表现出显著的行为差异，证明研究路线可行、数据可得。（详见"已有工作基础"。）

## 五、预期成果

1. **系统**：一套完整的四足机器人分层运动控制系统（仿真），含任务环境、双层训练管线、冻结协议评估器、训练监控与证据管理工具，全部代码开源可复现；
2. **数据与分析**：难度阶梯上分层与端到端架构的受控对比数据（成功率、样本效率、失败归因），多种子验证报告；
3. **论文**：符合学校要求的毕业设计论文一篇。

## 六、进度安排

| 阶段 | 时间 | 内容 |
| --- | --- | --- |
| 1 | 第 1–2 周 | 文献调研；Isaac Lab 环境与底层策略训练 |
| 2 | 第 3–4 周 | 任务环境设计实现；预注册评估协议；底层策略达标 |
| 3 | 第 5–6 周 | 分层/单层两配置实现；难度阶梯受控实验 |
| 4 | 第 7–8 周 | 多种子验证；失败归因分析；数据冻结 |
| 5 | 第 9–12 周 | 论文初稿撰写 |
| 6 | 第 13–14 周 | 修改定稿；答辩准备 |

[待补：按学校实际周次换算为具体日期]

## 七、已有工作基础

1. **训练环境**：Isaac Lab 2.7.0 + Isaac Sim 5.1 训练链路已打通并完成版本锁定；2048 并行环境实测吞吐约 6.1 万 env-steps/s，显存峰值 4.2 GB（RTX 5060 Ti 16 GB）；
2. **底层策略（S0）**：Go2 平地速度跟踪策略已完成 3 个训练种子，冻结种子评估速度跟踪误差 0.0569±0.0011 m/s（阈值 0.15），零摔倒；
3. **任务与评估**：四阶段命令序列环境已实现并修复关键设计缺陷（目标坐标系锚定、闭环阶段指令、漂移式停止判据）；冻结种子评估器、阶段到达率统计已运行；
4. **初步对比结果**：预实验显示，分层配置在四阶段任务全部训练种子上 100 迭代收敛（成功率 100%），单层配置 3 种子中 1 个收敛（450 迭代）、其余未收敛——结果方向与预期假设一致，且多种子差异本身构成有价值的研究发现；
5. **工程设施**：训练证据束与自动报告系统、收敛早停机制、实时监控面板、运动轨迹可视化均已建成，保证后续实验数据完整可追溯。

## 八、参考文献

[1] Hwangbo J, Lee J, Dosovitskiy A, et al. Learning agile and dynamic motor skills for legged robots[J]. Science Robotics, 2019, 4(26). [待补：卷期页码]
[2] Lee J, Hwangbo J, Wellhausen L, et al. Learning quadrupedal locomotion over challenging terrain[J]. Science Robotics, 2020, 5(47). [待补]
[3] Rudin N, Hoeller D, Reist P, et al. Learning to walk in minutes using massively parallel deep reinforcement learning[C]//Conference on Robot Learning (CoRL). PMLR, 2022: 91-100.
[4] Margolis G B, Agrawal P. Walk these ways: Tuning robot control for generalization with multiplicity of behavior[C]//CoRL. 2022. [待补]
[5] Makoviychuk V, Wawrzyniak L, Guo Y, et al. Isaac Gym: High performance GPU-based physics simulation for robot learning[EB/OL]. arXiv:2108.10470, 2021. [待核实]
[6] Sutton R S, Precup D, Singh S. Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning[J]. Artificial Intelligence, 1999, 112(1-2): 181-211.
[7] Bacon P L, Harb J, Precup D. The option-critic architecture[C]//AAAI. 2017. [待补]
[8] Vezhnevets A S, Osindero S, Schaul T, et al. FeUdal networks for hierarchical reinforcement learning[C]//ICML. 2017. [待补]
[9] Pateria S, Subagdja B, Tan A H, et al. Hierarchical reinforcement learning: A comprehensive survey[J]. ACM Computing Surveys, 2021, 54(5): 1-35. [待核实]
[10] Jain D, Iscen A, Caluwaerts K. Hierarchical reinforcement learning for quadruped locomotion[C]//IROS. 2019. [待补]
[11] Watanabe T, et al. Hierarchical reinforcement learning with central pattern generators for quadruped locomotion[J]. Scientific Reports, 2025. [待核实]
[12] Peng X B, Ma Y, Abbeel P, et al. AMP: Adversarial motion priors for stylized physics-based character control[J]. ACM TOG, 2021. [待补]
[13] Ma Y J, Liang W, Wang G, et al. Eureka: Human-level reward design via coding large language models[C]//ICLR. 2024. [待补]
[14] Ma Y J, et al. DrEureka: Language model guided sim-to-real transfer[C]//CoRL. 2024. [待补]
[15] Grandia R, Taylor A J, Perk E, et al. Multi-layered safety for legged robots via control barrier functions[EB/OL]. arXiv:2209.07522, 2022. [待核实]
[16] Zhang H, et al. Deep reinforcement learning for real-world quadrupedal locomotion: A comprehensive survey[J]. Intelligent Robotics, 2022. [待核实]

---

## 附：写作与交稿清单（不入正文）

- [ ] 按学校模板套版：封面/表格/字数/格式（GB/T 7714 引用格式逐条核对）
- [ ] ⚠️ 标记的 8 条文献逐条核实卷期页码
- [ ] 补 3–5 篇中文文献（知网：四足机器人 RL 运动控制、分层强化学习）
- [ ] 技术路线图绘制为正式插图（当前为文字框图）
- [ ] "已有工作基础"若学校要求附证明材料：附报告链接与截图
- [ ] 导师审阅后定稿
