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


def bootstrap() -> None:
    """Register verl-omni's adapters (rollout, model architecture, processor)
    with upstream verl.

    Idempotent and lightweight — no torch/vllm imports at the package top level.
    Call this explicitly from trainer entry points, tests, and the Ray
    ``worker_process_setup_hook`` instead of relying on import-time side effects.
    """
    from verl_omni.registry import register_all

    register_all()


def _init_worker() -> None:
    """Ray worker_process_setup_hook entry point."""
    bootstrap()
