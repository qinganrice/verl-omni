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


import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "version/version")) as f:
    __version__ = f.read().strip()


# Patch upstream verl: register vllm_omni rollout, Qwen3-Omni model class, etc.
# Lightweight — touches only verl/transformers metadata, no torch/vllm imports,
# so it is safe to run inside Ray worker_process_setup_hook before the actor
# has been pinned to a specific CUDA device.
import verl_omni._upstream_patches  # noqa: E402, F401


def bootstrap() -> None:
    """Driver-side bootstrap: import sub-modules to trigger their registrations.

    This is intentionally NOT done at top-level import time because
    ``verl_omni.workers.*`` transitively imports vllm/torch and would
    initialize CUDA in Ray worker processes before they have been narrowed
    to a single GPU via ``CUDA_VISIBLE_DEVICES`` — causing NCCL to see
    duplicate ranks on the same physical device.
    """
    import verl_omni.pipelines  # noqa: F401
    import verl_omni.reward_loop  # noqa: F401
    import verl_omni.workers.engine  # noqa: F401
    import verl_omni.workers.rollout  # noqa: F401


def _init_worker() -> None:
    """Ray ``worker_process_setup_hook`` entry point.

    Importing this module already applied the upstream patches above; this
    function exists so Ray can resolve ``verl_omni._init_worker``.
    """
    return None
