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
"""Upstream model patches — NOT model implementations.

Each subpackage applies, on import, the transformers/verl patches a specific
model needs to train under verl-omni (e.g. registering it with
``AutoModelForCausalLM`` or extending ``hf_processor``). To support a new model,
add a subpackage and import it below.
"""
from . import qwen3_omni_thinker  # noqa: F401  applies the model's patches on import
