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

import logging
import os
from contextlib import AbstractContextManager, contextmanager, nullcontext

import torch
from vllm.utils.mem_utils import GiB_bytes

__all__ = ["NPUColocateWorkerMixin"]


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def _is_npu_platform() -> bool:
    """Return True when vLLM is running on an Ascend NPU device."""
    try:
        from vllm.platforms import current_platform

        return current_platform.device_type == "npu"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NPU memory allocator
# ---------------------------------------------------------------------------


def _get_npu_memory_allocator():
    """Return the singleton CaMemAllocator instance for NPU memory pools."""
    from vllm_ascend.device_allocator.camem import CaMemAllocator

    return CaMemAllocator.get_instance()


# ---------------------------------------------------------------------------
# Context manager: suppress diffusers empty-cache calls on NPU
# ---------------------------------------------------------------------------


@contextmanager
def _skip_diffusers_npu_empty_cache():
    """Temporarily patch diffusers so that NPU empty-cache calls are skipped.

    On Ascend NPU, calling ``empty_device_cache`` while inside a CaMemAllocator
    memory pool invalidates the pool's internal bookkeeping.  This context
    manager monkey-patches the two relevant diffusers helpers for the duration
    of a ``with`` block and restores the originals on exit.
    """
    try:
        from diffusers.models import modeling_utils
        from diffusers.utils import torch_utils
    except Exception:
        yield
        return

    original_modeling_empty_cache = modeling_utils.empty_device_cache
    original_torch_empty_cache = torch_utils.empty_device_cache

    def empty_device_cache(device_type: str | None = None):
        if device_type is None or device_type == "npu":
            return
        return original_torch_empty_cache(device_type)

    modeling_utils.empty_device_cache = empty_device_cache
    torch_utils.empty_device_cache = empty_device_cache
    try:
        yield
    finally:
        modeling_utils.empty_device_cache = original_modeling_empty_cache
        torch_utils.empty_device_cache = original_torch_empty_cache


# ---------------------------------------------------------------------------
# Mixin: NPU-specific overrides for vLLMOmniColocateWorkerExtension
# ---------------------------------------------------------------------------


class NPUColocateWorkerMixin:
    """Mixin that *would* override memory-pool, sleep, and wake_up on Ascend NPU.

    Usage::

        class vLLMOmniColocateWorkerExtension(NPUColocateWorkerMixin, CustomPipelineWorkerExtension):
            ...

    NOTE: vLLM's ``worker_base.init_worker`` asserts that extension methods do
    NOT shadow methods already defined on the worker class. Since
    ``_maybe_get_memory_pool_context``, ``sleep`` and ``wake_up`` are all
    defined on ``vllm_omni.worker.base`` (and ``CustomPipelineWorkerExtension``
    inherits from it), defining them on the mixin triggers ``AssertionError``
    at worker init time even on CUDA where the overrides are intentional
    no-ops (they fall back to ``super()``).

    As a temporary workaround, the NPU-specific overrides are removed; the
    mixin is kept as an empty class so that
    ``vLLMOmniColocateWorkerExtension``'s class signature remains stable.
    NPU users currently lose NPU-specific sleep/wake_up — see the original
    implementation in git history if needed.

    # TODO (long): Once vLLM-Omni provides first-class NPU support in
    ``CustomPipelineWorkerExtension`` (or once vLLM relaxes the no-shadow
    assertion), this mixin can either be removed entirely or have its
    NPU-only overrides restored using a different mechanism (e.g. dynamic
    method registration only when ``_is_npu_platform()`` is True).
    """

    pass
