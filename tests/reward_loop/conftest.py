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
# These reward-loop tests exercise the diffusion stack; import the diffusion
# subpackages so their registrations (model adapters, engines, reward managers)
# are applied before the tests run.
import verl_omni.pipelines  # noqa: F401
import verl_omni.reward_loop  # noqa: F401
import verl_omni.workers.engine  # noqa: F401