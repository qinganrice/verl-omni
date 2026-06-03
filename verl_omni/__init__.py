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


# Register verl-omni with upstream verl/transformers. The patches are loaded via
# importlib inside ``_loader.apply`` (rather than hard-importing the patch
# modules here). This runs at ``import verl_omni`` time — before the trainer
# imports ``verl.utils`` — because the hf_processor wrapper must be installed
# before verl captures a reference to ``hf_processor``. Lightweight: no
# torch/vllm imports, so it is safe inside a Ray ``worker_process_setup_hook``.
from verl_omni._loader import apply as _apply_patches  # noqa: E402

_apply_patches()


def bootstrap() -> None:
    """Driver-side bootstrap: eager-import sub-modules to trigger their registrations.

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

    Importing this package already applied the upstream patches (see
    ``_apply_patches`` above). If verl_omni displaced a pre-existing setup hook
    (recorded by ``_loader`` in ``VERL_OMNI_CHAINED_SETUP_HOOK``), run it now so
    it is chained rather than silently dropped.
    """
    import os

    chained = os.environ.get("VERL_OMNI_CHAINED_SETUP_HOOK")
    if not chained:
        return
    import importlib

    module_name, _, attr = chained.rpartition(".")
    if not module_name:
        return
    getattr(importlib.import_module(module_name), attr)()
