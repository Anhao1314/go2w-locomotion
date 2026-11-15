"""协议常量：因子计算依赖的固定参数。

这些值来自预注册协议或实测配置，集中在一处，避免各脚本各写一份。
任何变更都必须同时更新注释里的出处。
"""

from __future__ import annotations

# 控制频率 50 Hz（Isaac / MuJoCo 两侧一致；evaluation.json 的 steps 都按它折算时间）
CONTROL_DT_S = 0.02

# 序列任务的位置容差（locomotion/sequence/stage_logic.py: POSITION_TOLERANCE_M）
POSITION_TOLERANCE_M = 0.30

# LIBERO 训练数据帧率（lerobot/libero meta/info.json: fps=10）
LIBERO_DATASET_FPS = 10
# LeRobot LiberoEnvConfig 默认控制频率（envs/configs.py: fps=20，作为 robosuite control_freq）
LIBERO_ENV_FPS = 20
