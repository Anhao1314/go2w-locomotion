#!/usr/bin/env bash
# Retry loop for the LIBERO SmolVLA fine-tune: the startup hub-metadata check
# is the only network dependency; transient timeouts are survived by retrying.
# Resume continues from the last checkpoint (020000, batch 32 semantics).
cd /mnt/d/work/go2w-locomotion
export HF_HUB_OFFLINE=0

for attempt in 6 7 8; do
  echo "=== attempt $attempt $(date -u +%H:%M:%S) ==="
  /root/vla-env/bin/lerobot-train \
    --config_path=/mnt/d/work/go2w-locomotion/logs/vla/libero_smolvla/checkpoints/last/pretrained_model/train_config.json \
    --resume=true \
    --batch_size=32 \
    --accelerator.gradient_accumulation.steps=2 \
    --output_dir=/mnt/d/work/go2w-locomotion/logs/vla/libero_smolvla \
    > logs/vla/train_libero_005.log 2>&1
  code=$?
  echo "=== attempt $attempt exit $code ==="
  if [ $code -eq 0 ]; then
    echo "=== TRAINING_COMPLETE ==="
    exit 0
  fi
  sleep 45
done
echo "=== ALL_ATTEMPTS_EXHAUSTED ==="
exit 1
