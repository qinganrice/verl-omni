# Qwen3-Omni Thinker GSPO + LoRA Quickstart

End-to-end RL fine-tuning of **Qwen3-Omni-30B-A3B Thinker** with **GSPO + LoRA**
on a math reasoning dataset, using FSDP for the actor and vLLM-Omni as the
async rollout backend.

This guide walks a fresh setup from zero hardware to first training step.

---

## 1. Hardware

- **4 × H100 80GB** (single node) — verified working baseline.
- The actor (FSDP, 30B + LoRA r=64, with param/optimizer offload) and the
  vLLM-Omni rollout (TP=4, `gpu_memory_utilization=0.2`) **colocate on the
  same 4 GPUs**. Each GPU peaks around 50–55 GiB during actor backward.
- Multi-node not yet validated — single-node configs only.

## 2. Software

You need three repos, all on specific branches:

| Repo | Branch | Purpose |
|---|---|---|
| [verl](https://github.com/verl-project/verl) | `main` (PR#6512) | RL training framework |
| [verl-omni](https://github.com/verl-project/verl-omni) | this PR | Qwen3-Omni glue + GSPO trainer entry point |
| [vllm-omni](https://github.com/vllm-project/vllm-omni) | `main` (PR#3915) | Inference / rollout backend |

Recommended install order (in a fresh Python 3.12 venv):

```bash
python -m venv .venv
source .venv/bin/activate

# 1. vllm + vllm-omni
pip install vllm==0.21.0 --torch-backend=auto
git clone https://github.com/vllm-project/vllm-omni.git
gh pr checkout 3915 #(change to PR (#3915))
pip install -e ./vllm-omni

# 2. verl (use the LoRA-FSDP-aware version)
git clone https://github.com/verl-project/verl.git
gh pr checkout 6512 #(change to PR (#6512))
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

## 3. Data

A parquet dataset of math problems with `prompt` and `answer` fields. The
script defaults to `~/data/math/{train,test}.parquet`. Any of the standard
RL math datasets work — the example was tested against `MATH-lighteval`.

To convert HuggingFace datasets to verl's parquet format, see
[`verl/examples/data_preprocess/`](https://github.com/verl-project/verl/tree/main/examples/data_preprocess).

Quick smoke run:

```bash
mkdir -p ~/data/math
# … place train.parquet and test.parquet here …
ls ~/data/math/  # should show train.parquet, test.parquet
```

## 4. Model

The script downloads `Qwen/Qwen3-Omni-30B-A3B-Instruct` from HuggingFace on
first launch (~60 GB). Two ways to pre-stage:

```bash
# Option A: HF CLI
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct

# Option B: point to a local copy
export MODEL_PATH=/path/to/local/Qwen3-Omni-30B-A3B-Instruct
```

> **Use the Instruct variant.** The base checkpoint does not ship a
> `tokenizer.chat_template` — verl's dataset loader calls
> `tokenizer.apply_chat_template(...)` and will fail without it.

## 5. Run

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

Configure W&B (optional):

```bash
export WANDB_API_KEY=...
# trainer.project_name / experiment_name are already set in the script
```

## 6. Toy test (quick pipeline verification)

Before committing to a 22-minute step, run a tiny config end-to-end to make
sure the whole pipeline works (rollout server starts, LoRA syncs, actor
back-propagates, validation runs). Random-initialize a 2-layer Qwen3-Omni
and save it locally so the model loads in seconds without the 60 GB
download:

```python
# build_dummy_qwen3_omni.py
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SRC = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DST = "./tiny-qwen3-omni"

config = AutoConfig.from_pretrained(SRC)
text_cfg = config.thinker_config.text_config
text_cfg.num_hidden_layers = 2          # 28 → 2
text_cfg.hidden_size = 256              # 2048 → 256
text_cfg.intermediate_size = 512
text_cfg.num_experts = 4                # 128 → 4 (MoE)
text_cfg.num_experts_per_tok = 2

model = AutoModelForCausalLM.from_config(config)
model.save_pretrained(DST)

# Copy tokenizer from the real checkpoint so chat_template is preserved.
AutoTokenizer.from_pretrained(SRC).save_pretrained(DST)
print(f"Dummy model saved to {DST}")
```

Run once:

```bash
python build_dummy_qwen3_omni.py
```

Then launch with `MODEL_PATH` pointing at the dummy:

```bash
MODEL_PATH=$PWD/tiny-qwen3-omni \
bash examples/gspo_trainer/run_qwen3_omni_thinker_gspo_lora.sh \
    data.train_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.val_max_samples=10 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.total_training_steps=2 \
    trainer.test_freq=1 \
    trainer.save_freq=999
```

Expect end-to-end (load → 2 train steps → validate → exit) in under 5
minutes. Loss values will be garbage — you're testing wiring, not
quality. If this passes, the full-size config is just slower, not
different in correctness.

## 7. What you should see

After ~22 minutes (one full step on 4×H100), the trainer logs a metrics
line. Healthy signals:

- `actor/loss` ≈ 1e-4 to 1e-3 (GSPO with tight clip ratio)
- `actor/grad_norm` between 1e-3 and 1
- `training/rollout_actor_probs_pearson_corr` > 0.95 (actor ↔ rollout
  agree on the policy distribution after weight sync)
- `actor/perf/max_memory_allocated_gb` < 60
- `val-core/.../acc/mean@1` rising with steps

## 8. What's actually being trained

Only the **Thinker** (Qwen3OmniMoeThinkerForConditionalGeneration), with:
- LoRA rank 64, alpha 32, on `target_modules="all-linear"`
- `exclude_modules` strips talker / code2wav / code_predictor / visual /
  audio_tower (we don't train those)
- `freeze_vision_tower=True` keeps even the vision encoder cold
- The other heads (talker etc.) are dropped at FSDP wrap time via
  `_verl_strip_modules`

Reward comes from the `dapo` reward manager (math accuracy on parsed
final answers).

## 9. Preliminary results

After a multi-step run with the default config above, validation accuracy
on MATH-lighteval lands around **0.886**. Training reward over steps:

![training reward](reward.png)

**Why no dramatic upward trend?** A few potential reasons:

- **Strong baseline.** Qwen3-Omni-30B-A3B-Instruct is already a strong
  zero-shot solver on MATH-lighteval, so the remaining gap is genuinely
  hard reasoning. RL on top of that ceiling moves slowly and noisily.
- **LoRA capacity ceiling.** rank 64 against a 30B base is enough to
  steer behaviour but not to overhaul it — useful for safety alignment
  and small skill nudges, less so for closing a hard reasoning gap.
- **Sparse, binary reward.** Math correctness is 0/1 with the `dapo`
  reward manager. With a high-baseline policy, most rollouts on a prompt
  share the same outcome, so advantages are small and gradient signal
  is dominated by variance.

Treat this run as a **plumbing-correctness signal** (loss finite, grad
norm reasonable, rollout↔actor pearson > 0.99, no OOM) — not as evidence
the recipe is optimal. To chase actual capability gains, expect to tune
clip ratios, learning rate, LoRA rank, and validate on a larger holdout.

---

## File map

```
examples/gspo_trainer/
├── run_qwen3_omni_thinker_gspo_lora.sh   ← launch script
├── qwen3_omni_thinker_only.yaml          ← vllm-omni stage config
├── reward.png                            ← preliminary reward curve
└── QuickStart.md                         ← (this file)
```
