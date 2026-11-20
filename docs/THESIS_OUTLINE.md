# 毕设论文骨架：章节 ↔ 仓库资产映射

> 用途：写作期的导航图。每章列出：本章回答什么问题、素材来源（文档/证据/图表）、
> 当前完备度。写正文时按此图取材，避免遗漏与重复。
> 数据一律以 [RESULTS.md](RESULTS.md)（单一事实源）为准；引用数字必须能在
> `reports/evidence/` 找到原始 JSON。

最后更新：2026-11-20

---

## 论文标题（开题已定）

**基于分层强化学习的四足机器人运动控制系统设计与仿真验证**

## 摘要要点（150–200 字模板）

端到端 RL 在四足单技能上成熟，但长程多阶段任务存在异质奖励与失败不可归因两个
困难；文献缺受控对照。本文设计 S2 规则层 + S1 指令策略 + S0 运动策略的分层系统，
并以同任务同观测同预算的单层系统为对照，在预注册协议（5 训练种子 × 20 冻结评估
种子）下完成：① 受控对比（分层 5/5 = 100% vs 单层 1/5，Fisher p = 0.048；收敛
100 迭代 vs 450/永不）；② 三臂归因消融（规则层直连即 100%——收益归因于动作抽象
+ 规则先验，非高层学习）；③ 零改动跨引擎验证（更严语义下 20/20）与精度轴（判据
收紧 6 倍不塌）；④ 鲁棒性三轴与反直觉发现。全部结论附证据链与证伪记录。

---

## 第 1 章 绪论

**回答**：为什么做这个题。

| 素材 | 来源 |
|---|---|
| 背景两段（并行 RL 成熟 / 长程任务两困难） | [proposal.md](proposal.md) §1（已起草）|
| 研究缺口（文献无受控对照） | proposal §1.2 + [RESEARCH.md](RESEARCH.md) §2 |
| 贡献列表 C1–C5 | [README.md](../README.md) §1.3（直接搬） |

**完备度**：🟢 90%（补：国内研究现状段落）

## 第 2 章 相关工作

**回答**：别人做到哪、缺口在哪。

| 主题线 | 素材 |
|---|---|
| 并行 RL 运动控制 | proposal §2.1（Hwangbo/Rudin/WTW，文献已列 16 篇）|
| 分层 RL 理论 | proposal §2.2（Options/Option-Critic/Feudal）|
| 具身记忆与目标条件（延伸线） | [MORA_RESEARCH.md](MORA_RESEARCH.md) §2026-11-20（MoRA 口径 + GIFT/Long-VLA/RL-SFT）|
| 缺口陈述 | RESEARCH.md §2.3 |

**完备度**：🟡 70%（补：MoRA 段落按 11-20 调研节重写；GIFT/Long-VLA 引用格式化）

## 第 3 章 系统设计（方法）

**回答**：系统怎么设计的、为什么这么设计。

| 小节 | 素材 |
|---|---|
| 3.1 分层架构（S2/S1/S0 + 对照单层） | README 系统组成图 + [ARCHITECTURE.md](ARCHITECTURE.md) |
| 3.2 任务设计（四阶段 + 判据 + 为什么平地盲感知） | [RESEARCH.md](RESEARCH.md) §4 任务设计 |
| 3.3 关键设计决策 | mdp/__init__.py 模块 docstring（闭环指令/漂移判据的失败修复史）|
| 3.4 评估协议（预注册/冻结种子/因子库/门禁） | [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) + [FACTORS.md](FACTORS.md) + [METHOD.md](METHOD.md) |

**完备度**：🟢 85%（补：把"设计决策的失败修复史"从代码注释提升为正文叙述）

## 第 4 章 受控对比实验（核心章）

**回答**：C1——分层 vs 端到端，统计显著吗。

| 小节 | 素材 |
|---|---|
| 4.1 实验设置（5 种子 × 20 冻结种子、协议） | [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) + 各 run 的 run_config.json |
| 4.2 主结果表（成功率/收敛迭代） | [l3_ab_comparison.md](../reports/training/l3_ab_comparison.md) §0（n=5 终稿）+ sweep 曲线图 |
| 4.3 统计检验（Fisher p=0.048 + 收敛一致性双支点） | l3_ab §0 结论 3 |
| 4.4 失败归因（四种死法 + 阶段到达率） | l3_ab §2 + [RESULTS.md](RESULTS.md) 行 1 |
| 4.5 方法论注脚（只跑单种子的两个错误结论） | l3_ab §0 注脚 |

**完备度**：🟢 95%（数据 n=5 全齐；写作即可）

## 第 5 章 归因消融（本文最有原创性的章）

**回答**：C2——收益归属哪一层。

| 小节 | 素材 |
|---|---|
| 5.1 三臂设计（Trained S1 / Oracle / Zero） | [RESULTS.md](RESULTS.md) §4.2 + mora 调研 §12 |
| 5.2 结果（规则直连 = 100%，S1 仅效率） | [RESULTS.md](RESULTS.md) 归因定案节 |
| 5.3 与 MoRA 内化主张的关系 | [MORA_RESEARCH.md](MORA_RESEARCH.md) §11–12 |
| 5.4 意义：接口设计 > 高层学习（在该任务类上） | Oracle 消融的讨论 |

**完备度**：🟢 90%（数据全齐；写作即可）

## 第 6 章 跨引擎验证与精度分析

**回答**：C3——结论跨引擎成立吗、判据虚不虚。

| 小节 | 素材 |
|---|---|
| 6.1 零改动迁移（ONNX + SHA-256） | [sim2sim/README.md](../locomotion/sim2sim/README.md) |
| 6.2 语义差异 caveat（0.5s vs 1s，移植更严） | sim2sim README 已知差异节 |
| 6.3 精度轴（容差 6× 扫描） | RESULTS 行 3b + [s4_tolerance_sweep.png](../media/analysis/s4_tolerance_sweep.png) |

**完备度**：🟢 90%

## 第 7 章 泛化与鲁棒性

**回答**：C4——没见过的场景行不行、扰动下稳不稳。

| 小节 | 素材 |
|---|---|
| 7.1 未见几何（80/80 vs 0/80 + 成本模型） | RESULTS 行 5–6 + generalization_cost 图 |
| 7.2 鲁棒性三轴（推力❌/载荷⚠️/摩擦✅） | RESULTS 行 7–8c + s2_failure_window 图 |
| 7.3 S2b 地形阶梯（实现 + 训练中） | [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) 执行状态表 + s2b_terrain_preview |

**完备度**：🟡 80%（S2b 训练完成后补 7.3 结果）

## 第 8 章 部署链路（ROS2）与延伸（VLA）

**回答**：系统能不能跑起来、思想能不能延伸。

| 小节 | 素材 |
|---|---|
| 8.1 ROS2 真 DDS 闭环（ROS-R1） | [go2_bridge README](../ros2/go2_bridge/README.md) + probe_receipt |
| 8.2 VLA 延伸：0.50 基线 + 七杠杆证伪 | [vla/README.md](../vla/README.md) + [RESULTS.md](RESULTS.md) §二 |
| 8.3 证伪的方法论价值 | RESULTS 未解问题节 + [METHOD.md](METHOD.md) |

**完备度**：🟢 85%

## 第 9 章 结论与展望

| 素材 |
|---|
| 结论三句（可靠性+效率、归因、跨引擎）——README 摘要改写 |
| 展望四条（地形/DR/内化/特征注入）——README §8 |
| 局限六条——README §7 |

**完备度**：🟢 90%

---

## 附录清单

| 附录 | 素材 |
|---|---|
| A 实验协议全文 | EXPERIMENT_DESIGN.md |
| B 因子库 | FACTORS.md |
| C 证据清单 | reports/evidence/INDEX.md |
| D 超参数与版本 | 各 run 的 run_config.json（依赖版本/命令行/git SHA）|

---

## 写作纪律（从本仓库方法论直接继承）

1. **每个数字必须能溯源**到 `reports/evidence/` 的 JSON——写作时逐条对照
   [RESULTS.md](RESULTS.md) 的证据列；
2. **统计表述用双支点**：Fisher p=0.048（成功率）+ 收敛迭代一致性（更强的支点）；
   不得只报 p 值；
3. **负结果照写**：S2 推力 0/20、技能条件化证伪、目标图像证伪——每个附机制解释；
4. **"MoRA-inspired"措辞**：对方无公开论文，不构成复现（MORA_RESEARCH.md §0 分级）；
5. **限定诚实**：平地盲感知主任务（归因需要）、n=5、单卡、无真机——写在局限章，
   不藏在脚注。

## 完备度总览

| 章 | 完备度 | 缺口 |
|---|---|---|
| 1 绪论 | 🟢 90% | 国内现状段 |
| 2 相关工作 | 🟡 70% | MoRA 段重写 + 新文献格式化 |
| 3 系统设计 | 🟢 85% | 决策史叙述 |
| 4 受控对比 | 🟢 95% | — |
| 5 归因消融 | 🟢 90% | — |
| 6 跨引擎 | 🟢 90% | — |
| 7 泛化鲁棒 | 🟡 80% | S2b 结果（训练中）|
| 8 部署+延伸 | 🟢 85% | — |
| 9 结论展望 | 🟢 90% | — |

**结论：数据侧 100% 就绪，写作侧约 85%——S2b 训练完成后即进入纯写作期。**
