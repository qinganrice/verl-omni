#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO + LoRA — IMAGE-input RL (phase 1, FineVideo frames MC).
# Reuses config/qwen3_omni_thinker_gspo.yaml and overrides only the multimodal
# bits (data, reward, prompt length, image stage config) on the CLI.
# Hardware: 4× H100 80GB.
set -x

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni.models.transformers.qwen3_omni_thinker

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/finevideo_img/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/finevideo_img/test.parquet"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIG="${SCRIPT_DIR}/qwen3_omni_thinker_only_image.yaml"
REWARD_FN="${SCRIPT_DIR}/mc_reward.py"

python3 -m verl.trainer.main_ppo \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=qwen3_omni_thinker_gspo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.image_key=images \
    data.max_prompt_length=2048 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_omni_thinker \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    reward.reward_manager.name=naive \
    custom_reward_function.path="${REWARD_FN}" \
    custom_reward_function.name=compute_score \
    trainer.experiment_name=gspo_lora_finevideo_image \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    "$@"
