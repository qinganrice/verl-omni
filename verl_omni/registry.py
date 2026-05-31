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
"""Register verl-omni's adapters with upstream verl.

Kept model-agnostic: the generic ``vllm_omni`` rollout adapter is registered
here, while model-specific patches (architecture, processor) are imported
lazily so heavy imports stay deferred until actually needed.
"""

_REGISTERED = False


def register_all() -> None:
    """Idempotently register all verl-omni adapters with upstream verl."""
    global _REGISTERED
    if _REGISTERED:
        return

    _register_vllm_omni_rollout()

    # Model-specific registrations. Lightweight — the patch module defers heavy
    # torch/transformers imports to lazy loaders inside each registration.
    import verl_omni.models.qwen3_omni_thinker._patches  # noqa: F401

    _REGISTERED = True


def _register_vllm_omni_rollout() -> None:
    """Register the (model-agnostic) vllm-omni async rollout adapter."""
    from verl.workers.rollout.base import register_rollout_adapter
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    register_rollout_adapter("vllm_omni", "async", "verl.workers.rollout.vllm_rollout.ServerAdapter")

    def _load_vllm_omni():
        from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

        return vLLMOmniReplica

    RolloutReplicaRegistry.register("vllm_omni", _load_vllm_omni)
