# 量化因子库（自动生成）

> 本文件由 `python analysis/extract.py` 生成，请勿手改；改定义请改 `analysis/factors.py`。

- 因子总数 **48**：已可采集 **44**，待补埋点 **4**
- 生成日期：2026-11-15
- 同一因子在所有脚本里只允许有一个实现（`analysis/factors.py`），避免口径漂移

## 速度跟踪（`tracking`）

| 因子 | 层级 | 单位 | 方向 | 阈值 | 定义 | 来源 |
|---|---|---|---|---|---|---|
| `track_vx_rmse` | rollout | m/s | 越小越好 | 0.15 | 前向速度跟踪均方根误差（相对指令） | evaluation.json: episodes[].vx_rmse |
| `track_wz_rmse` | rollout | rad/s | 越小越好 |  | 偏航角速度跟踪均方根误差 | evaluation.json: episodes[].wz_rmse |
| `track_vx_bias` | rollout | m/s | 看符号 |  | 实际平均速度减指令速度（有符号；负=系统性跟不上） | evaluation.json: mean_vx - mean_cmd_vx |
| `track_vx_rmse_cv` | run | 1 | 越小越好 |  | vx_rmse 的种子间变异系数（标准差/均值） | 派生 |
| `track_wz_rmse_cv` | run | 1 | 越小越好 |  | wz_rmse 的种子间变异系数 | 派生 |
| `track_axis_ratio_wz_vx` | run | 1 | 越小越好 |  | wz_rmse / vx_rmse：转向相对直行的跟踪难度倍数 | 派生 |

## 稳定性 / 扰动恢复（`stability`）

| 因子 | 层级 | 单位 | 方向 | 阈值 | 定义 | 来源 |
|---|---|---|---|---|---|---|
| `stab_fall` | rollout | 0/1 | 越小越好 |  | 该 episode 是否摔倒 | evaluation.json: episodes[].fall |
| `stab_fall_rate` | run | 1 | 越小越好 | 0.1 | 摔倒比例（门禁 ≤0.10） | 派生 |
| `stab_no_fall_rate` | run | 1 | 越大越好 | 0.9 | 无摔倒比例（S2 门禁 ≥0.90） | 派生 |
| `stab_recovery_time_s` | rollout | s | 越小越好 |  | 扰动后恢复到直立的时间（未恢复记 None） | evaluation.json: episodes[].recovery_time_s |
| `stab_fall_after_push_s` | rollout | s | 越小越好 |  | 推力施加后到摔倒的延迟（失效时间窗） | 派生：steps 差 × control_dt |
| `stab_max_tilt_deg` ⚠️ | rollout | deg | 越小越好 |  | 机身最大倾角（需要逐步姿态日志） | 未采集：需记录 root quaternion |
| `stab_slip_ratio` ⚠️ | rollout | 1 | 越小越好 |  | 接触滑动步数占比（需要逐步接触状态） | 未采集：需记录 contact |

## 任务完成（`task`）

| 因子 | 层级 | 单位 | 方向 | 阈值 | 定义 | 来源 |
|---|---|---|---|---|---|---|
| `task_success` | rollout | 0/1 | 越大越好 |  | 该 episode 是否判定成功 | evaluation.json: episodes[].success |
| `task_success_rate` | run | 1 | 越大越好 | 0.9 | 成功率（预注册门禁 ≥0.90） | 派生 |
| `task_steps` | rollout | steps | 越小越好 |  | 完成所需控制步数 | evaluation.json: episodes[].steps |
| `task_time_s` | rollout | s | 越小越好 |  | 完成所需时间 = 步数 × 控制周期 | 派生 |
| `task_terminal_miss_m` | rollout | m | 越小越好 |  | 终点到目标的平面距离 | evaluation.json: |terminal_pos - target_p2| |
| `task_miss_ratio` | rollout | 1 | 越小越好 |  | 终点残差 / 位置容差（成功预算的消耗比例） | 派生：miss / 0.30 |
| `task_stage_reached` | rollout | stage | 越大越好 |  | 达到的最大阶段（0-4） | evaluation.json: episodes[].max_stage |
| `task_steps_cv` | run | 1 | 越小越好 |  | 完成步数的变异系数（指标分辨率；0 = 测不出改进） | 派生 |
| `task_path_efficiency` ⚠️ | rollout | 1 | 越大越好 |  | 直线距离 / 实际路径长度（需要轨迹日志） | 未采集：需记录逐步位置 |
| `task_stage_time_s` ⚠️ | rollout | s | 越小越好 |  | 各阶段耗时（需要阶段切换时间戳） | 未采集：需记录 stage transition |

## 泛化（`generalization`）

| 因子 | 层级 | 单位 | 方向 | 阈值 | 定义 | 来源 |
|---|---|---|---|---|---|---|
| `gen_success_variant` | semantic | 1 | 越大越好 |  | 某几何变体上的成功率 | generalization: per_variant |
| `gen_steps_variant` | semantic | steps | 越小越好 |  | 某几何变体上的平均完成步数 | generalization: per_variant |
| `gen_fixed_overhead_steps` | run | steps | 越小越好 |  | 拟合的任务固定开销（起停 + 转向段），steps = a + b·distance | 派生 |
| `gen_per_metre_steps` | run | steps/m | 越小越好 |  | 每多 1 米路程增加的步数（拟合斜率） | 派生 |
| `gen_turn_cost_per_deg` | run | steps/deg | 越小越好 |  | 每多 1 度转向增加的步数 | 派生 |

## VLA 开环（`vla_openloop`）

| 因子 | 层级 | 单位 | 方向 | 阈值 | 定义 | 来源 |
|---|---|---|---|---|---|---|
| `vla_ol_mse` | run | action² | 越小越好 |  | 开环动作预测 MSE | openloop_summary.json: mse |
| `vla_ol_mse_ratio` | run | 1 | 越小越好 |  | MSE / 平均动作基线 MSE（<1 表示优于常数预测） | 派生 |
| `vla_ol_mae_dim` | semantic | action | 越小越好 |  | 逐维 MAE（7 维：3 位移 + 3 旋转 + 1 夹爪） | openloop records |
| `vla_ol_nmae_dim` | semantic | 1 | 越小越好 |  | 逐维归一化 MAE = MAE / GT 标准差（跨量纲可比） | 派生 |
| `vla_ol_bias_dim` | semantic | action | 看符号 |  | 逐维有符号偏置（区分系统性偏差与噪声） | 派生 |
| `vla_ol_corr_dim` | semantic | 1 | 越大越好 |  | 逐维预测-真值相关系数 | 派生 |
| `vla_ol_mae_progress` | semantic | action | 越小越好 |  | 按 episode 进度分箱的动作 MAE | 派生 |
| `vla_ol_mae_gripper_closed` | semantic | action | 越小越好 |  | 夹爪闭合（搬运）段的臂部 MAE | 派生 |
| `vla_ol_mae_gripper_open` | semantic | action | 越小越好 |  | 夹爪张开（接近/释放）段的臂部 MAE | 派生 |
| `vla_ol_gripper_sign_acc` | run | 1 | 越大越好 |  | 夹爪开合符号判断准确率 | 派生 |
| `vla_ol_vision_delta` | run | action | 越大越好 |  | 遮住图像后的动作偏移 L2（视觉依赖强度，太低=不看图） | 派生 |

## VLA 闭环（`vla_closed`）

| 因子 | 层级 | 单位 | 方向 | 阈值 | 定义 | 来源 |
|---|---|---|---|---|---|---|
| `vla_closed_success_rate` | run | 1 | 越大越好 |  | 闭环评测成功率（参考区间：LIBERO 文献 66–87%） | libero_eval_summary.json: success_rate |
| `vla_control_freq_ratio` | run | 1 | 看符号 |  | 环境控制频率 / 数据集元数据帧率；A/B 实测 2.0 有效（7/10）而 1.0 全败（0/10）→ 元数据 fps 不是控制契约 | analysis/protocol.py: LIBERO_ENV_FPS / LIBERO_DATASET_FPS |

## 训练效率（`training`）

| 因子 | 层级 | 单位 | 方向 | 阈值 | 定义 | 来源 |
|---|---|---|---|---|---|---|
| `train_smp_per_s` | train_step | samples/s | 越大越好 |  | 样本吞吐 | train log: smp/s |
| `train_step_s` | train_step | s | 越小越好 |  | 每优化步墙钟时间 | train log: step_s |
| `train_mem_gb` | train_step | GB | 越小越好 |  | 训练显存占用 | train log: mem_gb |
| `train_loss_decay_per_10k` | run | loss/10k steps | 越大越好 |  | 每 1 万步的 loss 降幅（收敛速率） | 派生 |
| `train_plateau_ratio` | run | 1 | 越小越好 |  | 后段收敛速率 / 前段收敛速率（越小越接近平台） | 派生 |
| `train_grad_norm` | train_step | 1 | 看符号 |  | 梯度范数 | train log: grdn |
| `train_lr_end` | run | 1 | 看符号 |  | 训练结束时的学习率（判断退火是否走完） | train log: lr |

## 待补埋点（`available=false`）

这些因子已定义但当前数据采不到。它们决定了下一轮实验应该记录什么：

| 因子 | 单位 | 需要的遥测 |
|---|---|---|
| `stab_max_tilt_deg` | deg | 未采集：需记录 root quaternion |
| `stab_slip_ratio` | 1 | 未采集：需记录 contact |
| `task_path_efficiency` | 1 | 未采集：需记录逐步位置 |
| `task_stage_time_s` | s | 未采集：需记录 stage transition |

## 当前 run 级因子快照

| run | n | `track_vx_rmse` | `track_wz_rmse` | `track_axis_ratio_wz_vx` | `stab_fall_rate` | `task_success_rate` | `task_steps` | `task_steps_cv` | `task_terminal_miss_m` |
|---|---|---|---|---|---|---|---|---|---|
| `s4_velocity_s0` | 20 | 0.07731 | 0.1391 | 1.799 | 0.05 | — | — | — | — |
| `s4_sequence_layered` | 20 | — | — | — | 0 | 1 | 474 | 0 | 0.2316 |
| `s4_sequence_single` | 20 | — | — | — | 1 | 0 | — | — | 2.484 |
| `gen_layered_trained_2m_90deg` | 20 | — | — | — | 0 | 1 | 474 | 0 | — |
| `gen_layered_unseen_2m_120deg` | 20 | — | — | — | 0 | 1 | 502 | 0 | — |
| `gen_layered_unseen_2m_60deg` | 20 | — | — | — | 0 | 1 | 447 | 0 | — |
| `gen_layered_unseen_3m_90deg` | 20 | — | — | — | 0 | 1 | 655 | 0 | — |
| `gen_single_trained_2m_90deg` | 20 | — | — | — | 1 | 0 | — | — | — |
| `gen_single_unseen_2m_120deg` | 20 | — | — | — | 1 | 0 | — | — | — |
| `gen_single_unseen_2m_60deg` | 20 | — | — | — | 1 | 0 | — | — | — |
| `gen_single_unseen_3m_90deg` | 20 | — | — | — | 1 | 0 | — | — | — |
| `s2_disturbance_push100N` | 20 | — | — | — | 1 | — | — | — | — |
| `baseline_c0_flat` | 20 | 0.6064 | — | — | 0 | — | — | — | — |
| `baseline_c0_push` | 20 | 0.5759 | — | — | 1 | — | — | — | — |
| `vla_closed_fps10` | 10 | — | — | — | — | 0 | — | — | — |
| `vla_closed_fps20` | 10 | — | — | — | — | 0.7 | 101.4 | 0.3303 | — |
| `vla_openloop_060k` | 40 | — | — | — | — | — | — | — | — |
| `gen_geometry_fit` | 4 | — | — | — | — | — | — | — | — |
| `vla_libero_setup` | 1 | — | — | — | — | — | — | — | — |
| `train_teacher_bs8` | 1 | — | — | — | — | — | — | — | — |
| `train_libero_bs8` | 1 | — | — | — | — | — | — | — | — |
| `train_libero_bs32` | 1 | — | — | — | — | — | — | — | — |

## 用法

```python
from analysis.factors import rollout_factors, run_factors, summarize, gate_check

rec = {'vx_rmse': 0.07, 'fall': False, 'success': True, 'steps': 474, 'control_dt': 0.02}
rollout_factors(rec)            # 单条 rollout → 因子
run_factors('my_run', [rec])    # 一组 rollout → run 级聚合
gate_check('track_vx_rmse', 0.0773)  # {'gate': 0.15, 'pass': True, 'margin': 1.94}
```

```bash
python analysis/extract.py       # 重建因子表与本文档
```
