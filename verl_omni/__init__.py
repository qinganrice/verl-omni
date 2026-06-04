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
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "version/version")) as f:
    __version__ = f.read().strip()


# Lightweight registration (no torch/vllm). verl discovers and imports this
# package via the "verl.plugins" entry point on the driver and inside Ray
# workers, so these registrations apply in every process.
def _register() -> None:
    from verl.workers.rollout.base import register_rollout_adapter
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    register_rollout_adapter("vllm_omni", "async", "verl.workers.rollout.vllm_rollout.ServerAdapter")

    def _load_vllm_omni():
        from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

        return vLLMOmniReplica

    RolloutReplicaRegistry.register("vllm_omni", _load_vllm_omni)

    import verl_omni.models  # noqa: F401  model patches self-register on import


_register()


def bootstrap() -> None:
    """Driver-side eager import of heavy (CUDA-touching) modules.

    Kept out of package import so ``import verl_omni`` stays lightweight and safe
    to run inside Ray workers before CUDA_VISIBLE_DEVICES is narrowed.
    """
    import verl_omni.pipelines  # noqa: F401
    import verl_omni.reward_loop  # noqa: F401
    import verl_omni.workers.engine  # noqa: F401
    import verl_omni.workers.rollout  # noqa: F401
