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
"""Thinker-only subclass of Qwen3OmniMoeForConditionalGeneration for RL training.

Overrides forward / embeddings to delegate directly to the thinker decoder,
sets FSDP split and strip lists, and suppresses tie_word_embeddings on the
config to avoid use_meta_tensor=False during FSDP init.
"""
from transformers.models.qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeForConditionalGeneration,
)


class _FalseTieDescriptor:
    # Forces tie_word_embeddings=False at the config class level so that verl's
    # FSDP init always uses use_meta_tensor=True (avoids OOM on 30B-A3B).
    def __get__(self, obj, objtype=None):
        return False

    def __set__(self, obj, value):
        pass


# Applied once at import time; safe because this module is only imported inside
# verl-omni's model registration path, never in a plain transformers context.
Qwen3OmniMoeConfig.tie_word_embeddings = _FalseTieDescriptor()


class Qwen3OmniThinkerOnlyModel(Qwen3OmniMoeForConditionalGeneration):
    """Qwen3-Omni wrapper that exposes only the thinker for FSDP/LoRA training."""

    # Corrects upstream typo: Qwen3OmniMoeDecoderLayer does not exist.
    _no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]

    # Heads not used during thinker-only training; dropped at FSDP wrap time.
    _verl_strip_modules = ["talker", "code2wav", "code_predictor"]

    def forward(
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

    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.thinker.set_input_embeddings(value)
