# 开题文献清单（按主题线组织）

> 用途：毕业设计开题报告的"国内外研究现状"素材库（PLAN.md T-开题）。
> 组织方式与开题综述的论述线一致。标注说明：✅ = 链接已核实；⚠️ = 来自检索摘要，引用前需核实页码/卷期。

## 线 1：大规模并行 RL 运动控制（本课题的直接技术基座）

| 文献 | 链接 | 与本课题的关系 |
| --- | --- | --- |
| Hwangbo, Lee et al. 2019. *Learning agile and dynamic motor skills for legged robots*. Science Robotics. (~2400 引用) | ⚠️ ETH Research Collection 有开放 PDF | RL+学习执行器模型做四足运动的奠基工作；引出"仿真训练+迁移"范式 |
| Lee, Hwangbo et al. 2020. *Learning quadrupedal locomotion over challenging terrain*. Science Robotics. (~2300 引用) | ⚠️ science.org | 提出 privileged learning / teacher-student 框架，是后续并行训练工作的基础 |
| Rudin et al. 2021. *Learning to Walk in Minutes Using Massively Parallel Deep RL*. CoRL 2021. | ✅ [arXiv:2109.11978](https://arxiv.org/abs/2109.11978) | 2048+ 并行环境 + 小 rollout（<25 步）+ 大 minibatch；本课题训练配置的直接来源 |
| Makoviychuk et al. 2021. *Isaac Gym: High Performance GPU-Based Physics Simulation*. | ⚠️ arXiv:2108.10470（引用前核实） | GPU 物理仿真平台，Isaac Lab 前身 |
| Margolis & Agrawal 2022. *Walk These Ways*. CoRL 2022. | ✅ [arXiv:2212.03238](https://arxiv.org/abs/2212.03238) | 行为参数化条件策略——本课题 S1 指令接口设计的备选方案来源 |

## 线 2：分层强化学习理论与方法（本课题的架构依据）

| 文献 | 链接 | 与本课题的关系 |
| --- | --- | --- |
| Sutton, Precup & Singh 1999. *Between MDPs and semi-MDPs*（Options 框架）. Artificial Intelligence. (~6100 引用) | ✅ [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370299000521) | 时间抽象与半马尔可夫决策过程——分层架构的理论起点 |
| Bacon, Harb & Precup 2017. *The Option-Critic Architecture*. AAAI. | ⚠️ [arXiv:1609.05140](https://arxiv.org/abs/1609.05140) | 端到端学习 option 的开创工作 |
| Vezhnevets et al. 2017. *FeUdal Networks for Hierarchical RL*. ICML. | ✅ [arXiv:1703.01161](https://arxiv.org/abs/1703.01161) | 经理-工人分层（manager 输出目标、worker 输出动作）——与本项目 S1/S0 结构同型 |
| Pateria et al. 2021. *Hierarchical RL: A Comprehensive Survey*. ACM CSUR. (~1000 引用) | ✅ [PDF](https://spateria.github.io/data/hrlsurvey_paper.pdf) | 综述：现状与开放问题，用于论证"缺受控对比"的研究缺口 |
| Jain et al. 2019. *Hierarchical RL for Quadruped Locomotion*. IROS 2019. | ✅ [arXiv:1905.08926](https://arxiv.org/abs/1905.08926) | 四足机器人上高层潜变量指令 + 低层执行的代表工作；但缺少同条件单层基线——正是本课题的切入点 |
| Watanabe et al. 2025. CPG + 分层策略的四足控制. Nature Sci. Reports. | ⚠️ [Nature](https://www.nature.com/articles/s41598-025-94163-2) | 近期分层四足工作（生物启发方向），用于现状部分的"近年进展" |

## 线 3：奖励设计与步态先验（本课题奖励迭代的对照参照）

| 文献 | 链接 | 与本课题的关系 |
| --- | --- | --- |
| Peng et al. 2021. *AMP: Adversarial Motion Priors*. SIGGRAPH. | ✅ [项目页](https://xbpeng.github.io/projects/AMP) | 用对抗先验替代手工奖励的原始工作 |
| Escontrela et al. 2022. *Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions*. IROS. | ✅ [arXiv:2203.15103](https://arxiv.org/abs/2203.15103) | 四足上 AMP 替代复杂奖励；本课题经历的三轮奖励迭代是手工奖励路线的实例，可在讨论中对照 |
| Ma et al. 2023. *Eureka: Human-Level Reward Design via Coding LLMs*. ICLR 2024. | ✅ [arXiv:2310.12931](https://arxiv.org/abs/2310.12931) | LLM 自动奖励设计（NVIDIA）；展望部分的素材 |
| Ma et al. 2024. *DrEureka: Language Model Guided Sim-To-Real*. CoRL 2024. | ✅ [arXiv:2406.01967](https://arxiv.org/abs/2406.01967) | LLM 自动域随机化 + 奖励，四足瑜伽球零样本迁移 |

## 线 4：安全与层间约束（架构 v0.2 子研究的理论支撑）

| 文献 | 链接 | 与本课题的关系 |
| --- | --- | --- |
| Grandia, Taylor et al. 2022. *Multi-Layered Safety for Legged Robots via Control Barrier Functions*. | ⚠️ [arXiv:2209.07522](https://arxiv.org/abs/2209.07522)（引用前核实） | 四足多层安全（CBF+MPC）——本课题"层间可验证约束"的同族工作 |
| Tan et al. 2022. *Safe RL for Legged Locomotion*. | ✅ [Google Research](https://research.google/pubs/safe-reinforcement-learning-for-legged-locomotion) | 安全恢复策略与学习策略切换 |
| Xiao et al. 2023. *BarrierNet: Differentiable CBFs*. IEEE T-RO. | ⚠️ [T-RO](https://ui.adsabs.harvard.edu/abs/2023ITRob..39.2289X/abstract) | 可微 CBF 嵌入网络——"约束式架构"谱系的现代形态 |

## 线 5：Sim-to-Real 与领域综述（背景与现状部分）

| 文献 | 链接 | 与本课题的关系 |
| --- | --- | --- |
| Tan et al. 2018. *Sim-to-Real: Learning Agile Locomotion*. RSS. (~1350 引用) | ✅ [RSS PDF](https://www.roboticsproceedings.org/rss14/p10.pdf) | 仿真迁移奠基工作（域随机化+系统辨识） |
| Zhao, Queralta et al. *Sim-to-Real Transfer in Deep RL for Robots: a survey*. | ⚠️ [Semantic Scholar](https://www.semanticscholar.org/paper/5a1b92aa50797a7c1e99b8840ff01aad66038596) | 域随机化/域适应/系统辨识三分类 |
| Zhang et al. 2022. *Deep RL for real-world quadrupedal locomotion: a comprehensive survey*. | ✅ [OAE](https://www.oaepublish.com/articles/ir.2022.20) | 四足 DRL 全景综述 |
| MDPI 2025. *Systematic Review of DRL for Legged Robot Locomotion*. | ⚠️ 检索到（PRISMA 综述，27 篇 2018–2025），引用前补全条目 | 最新系统综述：奖励塑形与训练稳定性影响 > 算法选择 |

## 线 6：算法与平台工具

| 文献/工具 | 链接 | 与本课题的关系 |
| --- | --- | --- |
| Schulman et al. 2017. *Proximal Policy Optimization Algorithms*. | ⚠️ [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)（引用前核实） | 本课题所用算法 |
| RSL-RL（Rudin 组维护的 PPO 库） | ✅ GitHub `leggedrobotics/rsl_rl` | 训练实现 |
| Isaac Lab（NVIDIA） | ✅ GitHub `isaac-sim/IsaacLab` | 仿真与任务框架 |
| legged_gym（ETH） | ✅ GitHub `leggedrobotics/legged_gym` | 线 1 工作的官方开源实现 |

## 论证主线（写综述时的逻辑串法）

1. **能力已解决**：并行 RL 让四足运动技能训练成为日常（线 1）；
2. **架构有传统**：分层 RL 有成熟理论与四足先例（线 2）；
3. **缺口**：先例各自报告成功率，**缺少同观测、同预算的端到端 vs 分层受控对比**，且失败发生时缺诊断手段；
4. **本课题**：受控 A/B 实验 + 难度阶梯 + 阶段级失败归因，补上这个缺口；
5. **延伸**：奖励工程与安全约束是两条相邻的活跃线（线 3、4），本课题的层间约束子研究与之接轨。

## 待办

- [ ] ⚠️ 项在正式引用前逐条核实（arXiv 号/卷期/页码）
- [ ] 中文文献补 3–5 篇（知网：四足机器人、强化学习运动控制）——开题报告通常要求中外文献混合
- [ ] 按学校模板的引用格式（GB/T 7714）排版
