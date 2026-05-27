# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Side-effect module: register verl-omni with upstream verl on import.

verl core deliberately does not depend on verl_omni. To keep that boundary
clean, every place where verl would need to know about ``vllm_omni`` /
Qwen3-Omni is patched here at import time. ``verl_omni/__init__.py`` imports
this module so any user that does ``import verl_omni`` (or runs
``verl_omni.trainer.omni.main_ppo``) automatically gets the registrations.
"""


# ---------------------------------------------------------------------------
# Rollout registry: tell verl how to spin up a vllm_omni rollout server.
# ---------------------------------------------------------------------------
def _register_vllm_omni_rollout() -> None:
    from verl.workers.rollout.base import _ROLLOUT_REGISTRY
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    # _ROLLOUT_REGISTRY maps (rollout_name, mode) -> ServerAdapter dotted path.
    # vllm_omni reuses verl's vLLM ServerAdapter for the HTTP layer.
    _ROLLOUT_REGISTRY.setdefault(
        ("vllm_omni", "async"),
        "verl.workers.rollout.vllm_rollout.ServerAdapter",
    )

    # RolloutReplicaRegistry maps rollout_name -> a callable that returns the
    # replica class. Lazy import so verl can be imported without verl_omni.
    def _load_vllm_omni():
        from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

        return vLLMOmniReplica

    RolloutReplicaRegistry.register("vllm_omni", _load_vllm_omni)


_register_vllm_omni_rollout()


# ---------------------------------------------------------------------------
# Qwen3-Omni Thinker: register with AutoModelForCausalLM and patch the few
# attributes that block FSDP-init / forward delegation.
# ---------------------------------------------------------------------------
def _register_qwen3_omni_automodel() -> None:
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_omni_moe import (
            Qwen3OmniMoeConfig,
            Qwen3OmniMoeForConditionalGeneration,
        )
    except ImportError:
        return

    from verl.utils.model import _architecture_to_auto_class

    # The thinker is decoder-only despite the "ForConditionalGeneration"
    # suffix; tell verl's architecture lookup to dispatch to AutoModelForCausalLM.
    _architecture_to_auto_class.setdefault(
        "Qwen3OmniMoeForConditionalGeneration", AutoModelForCausalLM
    )

    def _qwen3_omni_get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def _qwen3_omni_set_input_embeddings(self, value):
        self.thinker.set_input_embeddings(value)

    def _qwen3_omni_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    Qwen3OmniMoeForConditionalGeneration.forward = _qwen3_omni_forward
    Qwen3OmniMoeForConditionalGeneration.get_input_embeddings = _qwen3_omni_get_input_embeddings
    Qwen3OmniMoeForConditionalGeneration.set_input_embeddings = _qwen3_omni_set_input_embeddings
    # Upstream lists Qwen3OmniMoeDecoderLayer which does not exist; fix to the real class.
    Qwen3OmniMoeForConditionalGeneration._no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]
    # _verl_strip_modules is read by verl's FSDPEngine to delete unused sub-modules
    # (talker / code2wav / code_predictor are not needed for Thinker-only training).
    Qwen3OmniMoeForConditionalGeneration._verl_strip_modules = [
        "talker",
        "code2wav",
        "code_predictor",
    ]

    # tie_word_embeddings=True forces use_meta_tensor=False during FSDP init
    # which OOMs on 30B-A3B. Override at the config-class level via a no-op
    # descriptor so config __init__ assignments are tolerated.
    class _FalseTieDescriptor:
        def __get__(self, obj, objtype=None):
            return False

        def __set__(self, obj, value):
            pass

    Qwen3OmniMoeConfig.tie_word_embeddings = _FalseTieDescriptor()
    AutoModelForCausalLM.register(Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration)


_register_qwen3_omni_automodel()


# ---------------------------------------------------------------------------
# Wrap verl.utils.tokenizer.hf_processor so it also recognizes the Qwen3-Omni
# multimodal processor. The original uses a ``match`` block that cannot be
# extended at runtime; we install a wrapper that handles the Qwen3-Omni case
# when the original returns None.
# ---------------------------------------------------------------------------
def _patch_hf_processor_for_qwen3_omni() -> None:
    try:
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
    except ImportError:
        return

    import types

    import verl.utils.tokenizer as _vt

    _original_hf_processor = _vt.hf_processor

    def _patched_hf_processor(name_or_path, **kwargs):
        result = _original_hf_processor(name_or_path, **kwargs)
        if result is not None:
            return result

        # Original returned None — either it's a tokenizer-only model (fine)
        # or it failed because of an unsupported processor (maybe Qwen3-Omni).
        try:
            from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

            processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
            if isinstance(processor, PreTrainedTokenizerBase):
                return None
            if processor.__class__.__name__ != "Qwen3OmniMoeProcessor":
                return None

            config = AutoConfig.from_pretrained(name_or_path, **kwargs)
            # Token IDs / spatial_merge_size live on thinker_config, not the
            # top-level Qwen3OmniMoeConfig that AutoConfig returns.
            processor.config = config.thinker_config
            processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
            model_class = Qwen3OmniMoeThinkerForConditionalGeneration
            processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
            processor.get_llm_pos_ids_for_vision = types.MethodType(
                model_class.get_llm_pos_ids_for_vision, processor
            )
            return processor
        except Exception:
            return None

    _vt.hf_processor = _patched_hf_processor


_patch_hf_processor_for_qwen3_omni()


# ---------------------------------------------------------------------------
# Ray worker bootstrap: ensure these patches also run inside Ray actor /
# task processes. ``ray.init`` is monkey-patched to inject a
# ``worker_process_setup_hook`` that imports verl_omni at worker startup,
# which transitively re-runs every patch above.
# ---------------------------------------------------------------------------
def _ensure_workers_import_verl_omni() -> None:
    try:
        import ray
    except ImportError:
        return

    if getattr(ray.init, "_verl_omni_patched", False):
        return

    _original_init = ray.init
    _hook = "verl_omni._init_worker"

    def _patched_init(*args, **kwargs):
        runtime_env = kwargs.get("runtime_env") or {}
        existing_hook = runtime_env.get("worker_process_setup_hook") if hasattr(runtime_env, "get") else None
        if not existing_hook:
            try:
                runtime_env["worker_process_setup_hook"] = _hook
            except TypeError:
                runtime_env = {**dict(runtime_env), "worker_process_setup_hook": _hook}
            kwargs["runtime_env"] = runtime_env
        return _original_init(*args, **kwargs)
    _patched_init._verl_omni_patched = True
    ray.init = _patched_init


_ensure_workers_import_verl_omni()
