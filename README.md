# 分层强化学习让四足机器人长程任务从不可能变为可靠：一项受控对比研究

**基于分层强化学习的四足机器人运动控制系统设计与仿真验证**（毕业设计项目仓库）

[English](README.en.md) · [文档索引](docs/README.md) · [完整研究报告](docs/PHASE1_REPORT.md) · [实验协议](docs/EXPERIMENT_DESIGN.md) · [过程日志](docs/journal/)

---

## 一句话结果

> 在同一任务、同一观测、同一训练预算下，**四阶段长程任务上分层架构在全部 3 个训练种子以 100 迭代收敛到 100% 成功率；端到端单层仅 1/3 种子成功（450 迭代），其余 2 个种子 1500 迭代完全失败**。分层把训练成功率从 1/3 提到 3/3，收敛时间从"最好 450 迭代 / 最差永不"压缩到稳定的 100 迭代。在简单任务上两者都能成功，分层收敛快 3 倍。失败可归因：单层不收敛的种子全部死在"行走→原地转向"的步态切换处。

## 核心数据

### L3（难）：前进 2 m → 原地转 90° → 前进 2 m → 停止 1 s（每组 3 个训练种子）

| 训练种子 | 单层成功率 | 单层收敛 | 分层成功率 | 分层收敛 |
| --- | --- | --- | --- | --- |
| 0 | 0%（1500 迭代不收敛，全摔） | — | 100% | 100 迭代 |
| 1 | 0%（1500 迭代不收敛） | — | 100% | 100 迭代 |
| 2 | 100% | 450 迭代 | 100% | 100 迭代 |

> 方法论注脚：只跑 seed 2 会得出"单层也行只是慢 4.5 倍"，只跑 seed 0/1 会得出"单层完全不可行"——都错。多种子不是锦上添花，是结论正确性的前提。

### L1（易）：前进 1.2 m → 停止 0.5 s

| 指标 | 单层 | 分层 |
| --- | --- | --- |
| 成功率 | 100%（150 迭代收敛） | 100%（**50 迭代**收敛） |

### L1（易）：前进 1.2 m → 停止 0.5 s

| 指标 | 单层 | 分层 |
| --- | --- | --- |
| 成功率 | 100%（150 迭代收敛） | 100%（**50 迭代**收敛） |

**难度阶梯的含义**：简单任务上分层只是"更快"（效率优势）；任务复杂到需要切换运动模式时，分层变成"能 vs 不能"（成败差异）——且单层的失败被阶段到达率精确定位到"行走→转向"的步态切换，这是分层架构独有的诊断能力。

**样例对比视频**：[分层成功完成四阶段](media/training/robot_motion_l3_layered.mp4) · [单层在转向处摔倒](media/training/robot_motion_l3_single_fail.mp4) · [样本效率曲线](media/training/sweep_ab_comparison.png)

详细报告：[L3 对比](reports/training/l3_ab_comparison.md) · [L1 对比](reports/training/ab_comparison.md) · [M1 底层策略](reports/locomotion/m1_evaluation.md)

## 系统组成

```text
任务/阶段管理（规则层：闭环速度指令 + 阶段判定）
   │  [vx, wz] 指令
   ▼
S1 指令策略（PPO，2 维动作）──分层组──┐
                                      ├→ 12 维关节 → Unitree Go2（Isaac Sim 5.1）
S0 locomotion 策略（PPO，冻结复用）──┘
   ▲
48 维本体观测（M1 预训练：vx_rmse 0.057 m/s，三种子）
```

- **单层组**：53 维观测直接输出 12 维关节——与分层组完全相同的观测与任务信息
- **分层组**：S1 在 2 维指令空间学习，执行委托给预训练并冻结的 S0

## 工程设施（可复现性）

| 设施 | 说明 |
| --- | --- |
| 预注册协议 | 冻结评估种子 E={0..19}、指标与阈值先于训练冻结（[EXPERIMENT_DESIGN](docs/EXPERIMENT_DESIGN.md)） |
| 每 run 证据束 | `run_config.json`（git SHA/依赖版本/命令行）+ 全文件 SHA-256 清单，训练结束自动生成 |
| 训练报告 | 每次训练自动产出 Markdown 报告，含首末/极值/趋势与自动异常检测（[示例](reports/training/2026-10-24_01-18-00.md)） |
| 收敛即停 | 训练中每 50 迭代做一次 2048 环境确定性评估，连续达标自动早停并记录收敛点 |
| 实时面板 | 本地 Web 面板：训练进度、reward 曲线、GPU、快照回放、报告与视频入口 |
| 运动视频 | 记录刚体轨迹渲染 3D 骨架动画（成功与失败案例均可回放） |
| 过程日志 | [docs/journal/](docs/journal/)：按日记录目标、决策、**失败与修正**（含十余个真实 debug 案例） |

## 诚实边界

- **仿真限定**：无真机实验；结论适用于 Isaac Sim 5.1 仿真环境
- **S0 预训练成本**：分层组的 S0 来自约 1200 s 预训练；分层优势部分来自动作抽象与 S0 复用（未训练的 S1 已有 75% 成功率），而非 S1 本身学得更快——已在报告中归因
- **闭环指令含任务知识**：规则层的转向修正编码了部分任务先验——这正是分层设计的取舍，正文有讨论
- **每组 3 个训练种子**（S0 固定为 M1 seed 0，种子变量作用于上层策略训练）；更多种子与真机验证列为后续工作

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| [`locomotion/`](locomotion/README.md) | Phase-2 代码：任务环境、A/B 训练器、评估、动作映射 |
| [`mora/`](docs/MORA_TRAIN.md) | 分层运行时系统层（S2→S1→S0 契约、记忆、进度闭环） |
| [`cpp/`](cpp/README.md) | C++ 控制栈（FK/IK/MPC 数值验证）+ guardrail 运动学核心 |
| [`reports/`](reports/training/ab_comparison.md) | 训练报告、对比报告、扫描数据（全部自动生成） |
| [`docs/`](docs/README.md) | 协议、架构、路线图、每日日志、ADR 决策记录 |
| `scripts/` | 训练、评估、扫描、报告、面板、视频等入口 |
| `media/training/` | 关键视频与曲线（过程视频归档于 `media/archive/`） |

## Phase-1：Go2W 轮足导航（已完成归档）

本仓库此前完成了 Unitree Go2W 轮式四足的长程导航研究（MoRA-Train 第一个 adapter：
弯道 20/20、多段 A→B 20/20、岔路 40/40，C++ S0 数值验证），完整研究报告见
[docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md)。Phase-2 的分层对比方法与工程设施
均由该阶段演进而来。

## 复现

```powershell
$env:PYTHONPATH = "$PWD"; $env:OMNI_KIT_ACCEPT_EULA = "YES"

# 训练分层组（L3）
.venv311\Scripts\python.exe scripts\train_ab.py --mode layered `
    --s0-checkpoint <S0 checkpoint> --num_envs 2048 --max_iterations 1500 --seed 0

# 训练单层组（L3）
.venv311\Scripts\python.exe scripts\train_ab.py --mode single `
    --num_envs 2048 --max_iterations 1500 --seed 0

# 冻结种子评估
.venv311\Scripts\python.exe scripts\eval_policy.py --task Go2Loco-Sequence-v0 `
    --run-dir <run> [--layered --s0-checkpoint <S0>] --episodes 20
```

环境：Python 3.11 + Isaac Lab 2.7.0 + Isaac Sim 5.1.0.0 + RTX 5060 Ti（详见 [locomotion/README](locomotion/README.md) 的版本锁定表）。

## 许可与引用

代码与文档见 [LICENSE](LICENSE)。第三方组件许可见 [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)。
