#!/usr/bin/env bash
# Resume the LIBERO SmolVLA fine-tune from the newest checkpoint (last/).
# Written after the 2026-11-15 03:30 WSL VM shutdown killed the run at
# step ~41k of log 005 while checkpoint 060000 was already on disk.
# Stage 6 log file; attempts are sequential and resume is idempotent.
cd /mnt/d/work/go2w-locomotion
export HF_HUB_OFFLINE=0

for attempt in 1 2 3 4; do
  echo "=== attempt $attempt $(date -u +%H:%M:%S) ==="
  /root/vla-env/bin/lerobot-train \
    --config_path=/mnt/d/work/go2w-locomotion/logs/vla/libero_smolvla/checkpoints/last/pretrained_model/train_config.json \
    --resume=true \
    --batch_size=32 \
    --accelerator.gradient_accumulation.steps=2 \
    --output_dir=/mnt/d/work/go2w-locomotion/logs/vla/libero_smolvla \
    > logs/vla/train_libero_006.log 2>&1
  code=$?
  echo "=== attempt $attempt exit $code $(date -u +%H:%M:%S) ==="
  if [ $code -eq 0 ]; then
    echo "=== TRAINING_COMPLETE ==="
    exit 0
  fi
  sleep 45
done
echo "=== ALL_ATTEMPTS_EXHAUSTED ==="
exit 1
