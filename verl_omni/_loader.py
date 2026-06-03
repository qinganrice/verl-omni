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
"""Wire verl-omni into upstream verl without verl depending on verl_omni.

``apply()`` is called once from ``verl_omni/__init__.py`` at import time (which
runs before the trainer imports ``verl.utils``, a timing the hf_processor
wrapper relies on). It:

  1. registers the (model-agnostic) ``vllm_omni`` async rollout adapter via
     verl's public ``RolloutReplicaRegistry``,
  2. loads each model-specific patch module via :func:`importlib.import_module`
     (each applies its patches as an import side effect — we do not hard-import
     the patch modules anywhere), and
  3. installs the Ray worker bootstrap so the same patches run inside Ray
     worker processes.

Kept lightweight: every heavy import (torch / vllm / transformers / ray) is
deferred to inside the functions below.
"""

import importlib
import logging

logger = logging.getLogger(__name__)

# Model-specific patch modules, loaded lazily via importlib so new model
# patches can be added here without hard-importing them at package top level.
# Each module applies its patches as an import side effect.
_PATCH_MODULES = ("verl_omni.models.qwen3_omni_thinker._patches",)

_APPLIED = False


def apply() -> None:
    """Idempotently register verl-omni's adapters / patches with upstream verl."""
    global _APPLIED
    if _APPLIED:
        return

    _register_vllm_omni_rollout()
    for module in _PATCH_MODULES:
        importlib.import_module(module)
    _ensure_workers_import_verl_omni()

    _APPLIED = True


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


# ---------------------------------------------------------------------------
# Ray worker bootstrap: ensure the patches also run inside Ray actor / task
# processes. ``ray.init`` is wrapped to inject a ``worker_process_setup_hook``
# that imports verl_omni at worker startup, which re-runs ``apply()``.
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
        # Normalize to a mutable mapping we can read/update.
        if not hasattr(runtime_env, "get"):
            runtime_env = dict(runtime_env)

        existing_hook = runtime_env.get("worker_process_setup_hook")
        if existing_hook == _hook:
            # Our hook is already installed; nothing to do.
            return _original_init(*args, **kwargs)

        if existing_hook:
            # Do NOT silently drop a pre-existing hook (that would prevent it
            # from running on workers). Chain it: record it so ``_init_worker``
            # runs it after applying verl_omni's patches.
            env_vars = dict(runtime_env.get("env_vars") or {})
            if isinstance(existing_hook, str):
                env_vars.setdefault("VERL_OMNI_CHAINED_SETUP_HOOK", existing_hook)
            else:
                logger.warning(
                    "verl_omni: replacing a non-string worker_process_setup_hook "
                    "(%r); it cannot be chained and will not run on Ray workers.",
                    existing_hook,
                )
            runtime_env = {**dict(runtime_env), "env_vars": env_vars}

        try:
            runtime_env["worker_process_setup_hook"] = _hook
        except TypeError:
            runtime_env = {**dict(runtime_env), "worker_process_setup_hook": _hook}
        kwargs["runtime_env"] = runtime_env
        return _original_init(*args, **kwargs)

    _patched_init._verl_omni_patched = True
    ray.init = _patched_init
