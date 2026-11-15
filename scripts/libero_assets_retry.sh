#!/usr/bin/env bash
# Retry loop: LIBERO scene asset download (hf CDN flaky — resumes partial
# downloads each attempt until the env initializes cleanly).
cd /mnt/d/work/go2w-locomotion
export MUJOCO_GL=egl

for attempt in 1 2 3 4 5 6 7 8; do
  echo "=== asset attempt $attempt $(date -u +%H:%M:%S) ==="
  echo N | timeout 500 /root/vla-env/bin/python -u -c "
import sys; sys.path.insert(0, '.')
from lerobot.envs.factory import make_env
from lerobot.envs.configs import LiberoEnvConfig
cfg = LiberoEnvConfig(task='libero_spatial')
cfg.init_states = True
envs = make_env(cfg)
env = envs['libero_spatial'][0]
obs, info = env.reset(seed=[0])
print('ENV_READY obs keys:', sorted(obs.keys()))
"
  code=$?
  echo "=== attempt $attempt exit $code ==="
  if [ $code -eq 0 ]; then
    echo "=== ASSETS_READY ==="
    exit 0
  fi
  sleep 30
done
echo "=== ASSETS_EXHAUSTED ==="
exit 1
