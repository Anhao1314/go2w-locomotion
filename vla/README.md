# vla/ — SmolVLA 微调与验评链

VLA（视觉-语言-动作）路线的代码与证据入口。模型：**SmolVLA 450M**（SigLIP 视觉 + SmolLM2 +
流匹配动作专家），训练/评测栈：LeRobot 0.7。

当前结论一句话：**闭环评测已跑通，`libero_spatial` 5 个任务 × 20 种子 = 100 集，合计 50/100（0.50，95% CI 0.40–0.60）**。
逐条数字见 [docs/RESULTS.md](../docs/RESULTS.md)，原始证据见
[reports/evidence/vla-2026-11-15/](../reports/evidence/vla-2026-11-15/)。

[返回首页](../README.md) · [结果总表](../docs/RESULTS.md) · [因子库](../docs/FACTORS.md) · [S4 Sim2Sim](../locomotion/sim2sim/README.md)

## 两条路线（第一条未走通，第二条进行中）

```text
路线 A：真机教师数据 → 仿真孪生（跨域）
  lerobot/svla_so101_pickplace（50 段人类遥操作，front+wrist 相机）
        ▼  SmolVLA 微调 20 000 步（batch 8）→ final loss 0.112
  在 so101-tabletop-mujoco 同构任务上评测
        ▼  1 个 episode / 3 600 步 → 放置 0/4 方块（sim/real gap，n=1 仅作定性）

路线 B：LIBERO 自洽（训练分布 = 评测分布）
  lerobot/libero（1 693 段 / 273 465 帧 / 元数据标 10 Hz / 256×256）
        ▼  SmolVLA 微调（batch 32 × accum 2，目标 100 000 步）
  62 000 步 checkpoint（loss 0.380，lr 已退火到 2.5e-6）
        ▼  开环复现检查：MSE 0.0656 vs 基线 0.2117（0.31×）— 通过
        ▼  闭环评测（libero_spatial，14 s 预算）
             fps=20 → 5 任务 × 20 种子 = 50/100 = 0.50（CI 0.40–0.60）  ← 有效配置
             fps=10 → 0/10（全部超时）
```

## 闭环评测踩过的坑（值得记住）

第一版评测把 robosuite 的原始观测**直接**喂给了策略：

| 项目 | 训练时 | 评测时（错） | 修复 |
| --- | --- | --- | --- |
| 图像 | float32 **CHW**，0–1，256×256 | uint8 **HWC**，0–255，`(1,360,360,3)` | uint8→float、HWC→CHW、去 batch 维 |
| 状态 | `(8,)` | `(1,8)` | squeeze |
| 任务 | `str` | `tuple('pick up ...',)` | 取首元素 |

结果：同一个 checkpoint 从 **0/3** 变成 **7/10**。**模型没错，接口错了。**
这也是为什么 `scripts/probe_libero_obs.py` 现在作为评测前的必跑探针保留。

## 模块

| 文件 | 职责 |
| --- | --- |
| `env.py` | 环境封装：SO-101 EnvHub 包装（指尖标定、特权几何）+ `make_libero_env` |
| `expert.py` | 脚本专家：9 阶段抓放状态机 + DLS IK（WIP，未达可用成功率） |
| `evaluate.py` | 教师数据策略在 SO-101 仿真上的评测（`select_action` 闭环） |
| `evaluate_libero.py` | LIBERO 评测（`lerobot` 环境 + 官方预处理管线） |
| `openloop_check.py` | **开环复现检查**：喂训练帧，比对预测动作与演示动作（诊断用） |

相关脚本：`scripts/vla_smoke.py`（环境冒烟）、`scripts/vla_view.py`（可视化）、
`scripts/libero_*_retry.sh`（网络抖动下的下载/训练重试）。

## 数据事实（本机缓存）

| 数据集 | 规模 | 规格 | 缓存体积 |
| --- | --- | --- | --- |
| `lerobot/libero` | 1 693 段 / 273 465 帧 | 10 Hz，256×256 双相机，state 8 维，action 7 维 | 947 MB（blobs） |
| `lerobot/svla_so101_pickplace` | 50 段 | 真机遥操作，front + wrist 相机 | 83 MB |
| LIBERO 场景资产 | 5 个 suite | robosuite 场景与贴图 | 403 MB |

模型：教师数据 checkpoint 20K（13 GB）、LIBERO checkpoint 10K–60K（7.4 GB），均在
`logs/vla/` 下（不入库，按 `train_config.json` 可重建）。

## 已知问题

| 问题 | 说明 |
| --- | --- |
| 结果覆盖面 | 0.50 覆盖 `libero_spatial` 5/10 个任务，不外推到其他 suite |
| **目标图像拼接已证伪** | GIFT-lite 简化版（帧拼接）使工作视野减半 → 1/40；GIFT 正版需特征注入架构，超预算 |
| **技能条件化已证伪** | 逐帧技能前缀训练（5K 步）无害但推理条件化全变体有害（0.10–0.20 vs 裸跑 0.40）：条件化自身行为=反馈环；详见 [RESULTS.md](../docs/RESULTS.md) 与 `reports/evidence/vla-2026-11-18/` |
| **逐种子不可复现** | 策略动作由 flow matching 采样，GPU 数值非确定性会在闭环里放大：seed 1003 两次运行分别 70 步成功 / 280 步超时。单批 10 集只能给出 0.30–0.70 的波动，必须按批次重复测量 |
| 控制频率必须显式指定 | 数据集元数据 `fps=10`，但实测 `fps=20` 才有效（0.70 vs 0.00）→ 评测固定传 `--fps 20` |
| 评测链路曾长期静默失败 | 观测格式错误不会报错，只会让策略"看起来不会做事"；探针脚本 `scripts/probe_libero_obs.py` 用于事前拦截 |
| 跨域证据样本量 | 只有 1 个 episode，不能当作成功率结论 |
| 自建专家不可用 | 状态机 + IK 在仿真里未达到可用成功率，保留为 WIP 并如实记录 |

## 诚实边界

- 全部为仿真内数据与仿真内评测，无真机实验声明
- 教师数据来自公开数据集的人类遥操作，非本仓库采集
- 开环检查只证明"策略能复现训练演示"，**不能**替代闭环成功率
- 训练配置、checkpoint 与日志以 `logs/vla/` 为准；进入结论的数字另存
  [reports/evidence/](../reports/evidence/) 并登记在 [docs/RESULTS.md](../docs/RESULTS.md)
