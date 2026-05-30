# Qwen3-Omni Thinker GSPO + LoRA

End-to-end RL fine-tuning of **Qwen3-Omni-30B-A3B Thinker** with **GSPO + LoRA**
on a math reasoning dataset, using FSDP for the actor and vLLM-Omni as the
async rollout backend.

## Prerequisites

**Hardware:** 4 × H100 80GB (single node). The actor (FSDP, 30B + LoRA r=64,
with param/optimizer offload) and the vLLM-Omni rollout (TP=4,
`gpu_memory_utilization=0.2`) colocate on the same 4 GPUs. Each GPU peaks
around 50–55 GiB during actor backward. Multi-node is not yet validated.

**Software:** Install the three required packages in a fresh Python 3.12 venv:

```bash
python -m venv .venv
source .venv/bin/activate

# 1. vllm + vllm-omni
pip install vllm==0.21.0 --torch-backend=auto
git clone https://github.com/vllm-project/vllm-omni.git
gh pr checkout 3915
pip install -e ./vllm-omni

# 2. verl
git clone https://github.com/verl-project/verl.git
gh pr checkout 6512
pip install -e ./verl

# 3. verl-omni
git clone https://github.com/verl-project/verl-omni.git
pip install -e ./verl-omni
```

Verify:

```bash
python -c "import verl, verl_omni, vllm, vllm_omni; print('OK')"
```

> **Note on numpy**: vllm 0.21+ pulls `numpy>=2.x`. verl/verl-omni still pin
> `numpy<2.0.0` in their setup, but the codepaths used here are compatible
> with numpy 2.x — pip will warn, you can ignore.

## Prepare the dataset

A parquet dataset with `prompt` and `answer` fields. The script defaults to
`~/data/math/{train,test}.parquet`. The example was tested against
`MATH-lighteval`.

To convert HuggingFace datasets to verl's parquet format, see
[`verl/examples/data_preprocess/`](https://github.com/verl-project/verl/tree/main/examples/data_preprocess).

```bash
mkdir -p ~/data/math
# place train.parquet and test.parquet here
```

## Prepare the model

The script downloads `Qwen/Qwen3-Omni-30B-A3B-Instruct` from HuggingFace on
first launch (~60 GB). To pre-stage:

```bash
# Option A: HF CLI
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct

# Option B: point to a local copy
export MODEL_PATH=/path/to/local/Qwen3-Omni-30B-A3B-Instruct
```

> **Use the Instruct variant.** The base checkpoint does not ship a
> `tokenizer.chat_template` — verl's dataset loader calls
> `tokenizer.apply_chat_template(...)` and will fail without it.

## Run training

```bash
cd verl-omni
bash examples/gspo_trainer/run_qwen3_omni_thinker_gspo_lora.sh
```

Override defaults via env vars or extra Hydra args:

```bash
MODEL_PATH=/local/Qwen3-Omni-30B-A3B-Instruct \
TRAIN_FILE=/data/custom_train.parquet \
bash examples/gspo_trainer/run_qwen3_omni_thinker_gspo_lora.sh \
    trainer.total_epochs=10 \
    actor_rollout_ref.actor.optim.lr=2e-6
```

## Logging

W&B logging is optional. Set `WANDB_API_KEY` before launching:

```bash
export WANDB_API_KEY=...
# trainer.project_name / experiment_name are already set in the script
```

## Expected output

After ~20 minutes (one full step on 4×H100), the trainer logs a metrics line.
Healthy signals:

- `actor/loss` ≈ 1e-4 to 1e-3 (GSPO with tight clip ratio)
- `actor/grad_norm` between 1e-3 and 1
- `training/rollout_actor_probs_pearson_corr` > 0.95 (actor ↔ rollout agree
  on the policy distribution after weight sync)
- `actor/perf/max_memory_allocated_gb` < 60
- `val-core/.../acc/mean@1` rising with steps

## What's actually being trained

Only the **Thinker** (Qwen3OmniMoeThinkerForConditionalGeneration), with:

- LoRA rank 64, alpha 32, on `target_modules="all-linear"`
- `exclude_modules` strips talker / code2wav / code_predictor / visual /
  audio_tower (we don't train those)
- `freeze_vision_tower=True` keeps the vision encoder cold
- The other heads are dropped at FSDP wrap time via `_verl_strip_modules`

Reward comes from the `dapo` reward manager (math accuracy on parsed final
answers).

## Preliminary results

After a multi-step run with the default config, validation accuracy on
MATH-lighteval lands around **0.886**. Training reward over steps:

![training reward](reward.png)

**Why no dramatic upward trend?** Qwen3-Omni-30B-A3B-Instruct is already a
strong zero-shot solver on MATH-lighteval, so the remaining gap is genuinely
hard reasoning. LoRA rank 64 is sufficient to steer behaviour but not to
overhaul it, and sparse binary reward means most rollouts on a prompt share
the same outcome, so gradient signal is dominated by variance. Treat this run
as a plumbing-correctness signal — not as evidence the recipe is optimal. To
chase actual capability gains, expect to tune clip ratios, learning rate, LoRA
rank, and validate on a larger holdout.

## Smoke test

To verify the full pipeline without the ~60 GB download, use the smoke test
in `tests/special_e2e/`:

```bash
bash tests/special_e2e/run_gspo_thinker_smoke.sh
```

This builds a tiny random-weight model and runs 2 training steps end-to-end.
Expect completion in under 5 minutes. Loss values will be garbage — you are
testing wiring, not quality.
