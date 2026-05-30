#!/usr/bin/env bash
# GSPO + LoRA Thinker e2e smoke test (minimal runtime).
#
# Builds a tiny random-weight Qwen3-Omni model, then runs 2 training steps
# end-to-end to verify: rollout server starts, LoRA syncs, actor
# back-propagates, and validation runs without errors.
#
# Requires: verl, verl-omni, vllm-omni installed.
#   dummy model built at MODEL_PATH (created by this script if missing)
#
# Override via env: NUM_GPUS, MODEL_PATH, DATA_DIR, TOTAL_TRAIN_STEPS
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen3-Omni}
DATA_DIR=${DATA_DIR:-${HOME}/data/math}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── Build dummy model if not present ──────────────────────────────────────────
if [ ! -d "${MODEL_PATH}" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_qwen3_omni.py" \
        --output_dir "${MODEL_PATH}"
fi

# ── Run training ──────────────────────────────────────────────────────────────
python3 -m verl_omni.trainer.omni.main_ppo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.val_max_samples=10 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${NUM_GPUS}" \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    trainer.test_freq=1 \
    trainer.save_freq=999
