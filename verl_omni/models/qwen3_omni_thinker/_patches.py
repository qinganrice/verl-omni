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
"""Register verl-omni's Qwen3-Omni Thinker adapter with upstream verl.

Imported at verl_omni package init time (lightweight — no torch/vllm).
All registrations use lazy loaders so heavy imports are deferred until
the registered factories are actually invoked.
"""


def _register_vllm_omni_rollout() -> None:
    from verl.workers.rollout.base import register_rollout_adapter
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    register_rollout_adapter("vllm_omni", "async", "verl.workers.rollout.vllm_rollout.ServerAdapter")

    def _load_vllm_omni():
        from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

        return vLLMOmniReplica

    RolloutReplicaRegistry.register("vllm_omni", _load_vllm_omni)


_register_vllm_omni_rollout()


def _register_qwen3_omni_automodel() -> None:
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeConfig
    except ImportError:
        return

    from verl.utils.model import register_model_architecture
    from verl_omni.models.qwen3_omni_thinker.model import Qwen3OmniThinkerOnlyModel

    register_model_architecture("Qwen3OmniMoeForConditionalGeneration", AutoModelForCausalLM)
    AutoModelForCausalLM.register(Qwen3OmniMoeConfig, Qwen3OmniThinkerOnlyModel)


_register_qwen3_omni_automodel()


def _register_qwen3_omni_processor() -> None:
    try:
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
    except ImportError:
        return

    import types

    from verl.utils.tokenizer import register_processor_handler

    def _qwen3_omni_processor_handler(name_or_path, processor, config, **kwargs):
        # Token IDs / spatial_merge_size live on thinker_config, not the top-level config.
        processor.config = config.thinker_config
        processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
        model_class = Qwen3OmniMoeThinkerForConditionalGeneration
        processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
        processor.get_llm_pos_ids_for_vision = types.MethodType(
            model_class.get_llm_pos_ids_for_vision, processor
        )
        return processor

    register_processor_handler("Qwen3OmniMoeProcessor", _qwen3_omni_processor_handler)


_register_qwen3_omni_processor()
