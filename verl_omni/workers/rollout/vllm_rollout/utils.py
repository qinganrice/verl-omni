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

import torch
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH, set_death_signal
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

from verl_omni.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack
from verl_omni.workers.rollout.vllm_rollout.npu_utils import NPUColocateWorkerMixin

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMOmniColocateWorkerExtension(NPUColocateWorkerMixin, CustomPipelineWorkerExtension):
    """
    The class for vLLM-Omni's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. NPU (Ascend) memory-pool, sleep, and wake_up — via NPUColocateWorkerMixin
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMOmniHijack.hijack()

        return super().__new__(cls)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            # === DIAGNOSTIC DUMP: confirm what arrived at rollout side ===
            # Compare with actor-side dump in verl/utils/fsdp_utils.py to
            # detect tensors lost / renamed during IPC transfer.
            try:
                import re
                from collections import defaultdict
                names = [n for n, _ in weights]
                per_expert = defaultdict(set)
                non_expert = 0
                for n in names:
                    m = re.search(r"layers\.(\d+)\..*experts\.(\d+)\.(\w+?)_proj\.lora_([AB])", n)
                    if m:
                        per_expert[(m.group(1), m.group(2))].add((m.group(3), m.group(4)))
                    else:
                        non_expert += 1
                need = {(p, ab) for p in ("gate", "up", "down") for ab in ("A", "B")}
                incomplete = {k: sorted(need - v) for k, v in per_expert.items() if v != need}
                logger.warning(
                    "[lora rollout dump] total=%d non_expert=%d experts=%d incomplete=%d",
                    len(names), non_expert, len(per_expert), len(incomplete),
                )
                if incomplete:
                    logger.warning("[lora rollout dump] incomplete sample: %s",
                                   list(incomplete.items())[:10])
            except Exception as _e:
                logger.warning("[lora rollout dump] skipped: %s", _e)
            # === END DIAGNOSTIC ===

            # Pass LoRA tensors as-is. The actor trains the full model
            # (Qwen3OmniMoeForConditionalGeneration) so weight names contain
            # "thinker.model." prefixes. parse_fine_tuned_lora_name applies
            # the thinker-only model's hf_to_vllm_mapper which converts
            # "thinker.model." → "language_model.model." correctly. Stripping
            # the prefix here would break that mapping.
            lora_request = OmniTensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=dict(weights),
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM-Omni load weights, loaded_params: {len(weights)}")
        else:
            logger.info("Loading standard weights (async)")
            self.load_weights(weights)

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication.

        Uses Ray job id + replica_rank + local_rank to form the handle so it
        matches the sender side regardless of CUDA_VISIBLE_DEVICES differences,
        avoids collisions when multiple replicas share the same node, and is
        unique per Ray job to avoid cross-job collisions on shared hosts. The
        job id is forwarded by the vLLMHttpServer actor as VERL_RAY_JOB_ID and
        inherited by this vLLM worker subprocess.
        """
        job_id = os.environ.get("VERL_RAY_JOB_ID", "0")
        replica_rank = os.environ.get("VERL_REPLICA_RANK", "0")
        return f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{self.local_rank}.sock"
