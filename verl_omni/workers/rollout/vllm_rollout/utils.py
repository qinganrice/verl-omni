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

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Conditionally include NPUColocateWorkerMixin in the class hierarchy.
# vLLM v1 multiproc_executor asserts that an extension does not redefine
# attributes already present on the worker class. On GPU,
# GPUARWorker already provides ``_maybe_get_memory_pool_context``, ``sleep``,
# and ``wake_up``, so adding the NPU mixin (which redefines them, even when
# guarded by ``_is_npu_platform()`` internally) trips that assertion. On NPU
# the mixin is required because the underlying worker does not implement
# the NPU-specific memory-pool / sleep / wake_up flow.
def _platform_extension_bases():
    # TODO: the NPU (Ascend) path below is not yet verified on real NPU hardware;
    #       only the GPU branch is exercised by current tests / training runs.
    try:
        from vllm.platforms import current_platform

        if current_platform.device_type == "npu":
            from verl_omni.workers.rollout.vllm_rollout.npu_utils import NPUColocateWorkerMixin

            return (NPUColocateWorkerMixin, CustomPipelineWorkerExtension)
    except Exception:
        pass
    return (CustomPipelineWorkerExtension,)


class vLLMOmniColocateWorkerExtension(*_platform_extension_bases()):
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

        # vllm's add_lora -> pack_moe requires all per-expert LoRA tensors at
        # once; bucketed transfer can split them, so accumulate every bucket
        # then call add_lora once. Non-LoRA loading streams per-bucket.
        if peft_config and base_sync_done:
            accumulated_weights: list[tuple[str, torch.Tensor]] = []

            def _accumulate(weights: list[tuple[str, torch.Tensor]]) -> None:
                accumulated_weights.extend(weights)

            receiver.receive_weights(on_bucket_received=_accumulate)
            try:
                self._update_weights(
                    accumulated_weights, peft_config=peft_config, base_sync_done=base_sync_done
                )
            finally:
                # Release accumulated LoRA tensors before the engine's next
                # wake_up; otherwise cloned buckets stay on GPU until Python
                # GC and wake_up's cumem remap can OOM.
                accumulated_weights.clear()
                del accumulated_weights
                import gc as _gc

                _gc.collect()
                torch.cuda.empty_cache()
        else:
            receiver.receive_weights(
                on_bucket_received=lambda weights: self._update_weights(
                    weights, peft_config=peft_config, base_sync_done=base_sync_done
                )
            )

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            lora_request = OmniTensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=dict(weights),
            )
            self.add_lora(lora_request)
            # Drop in-memory tensor ref so vLLM's active-LoRA registry doesn't pin the adapter on GPU.
            lora_request.lora_tensors = None
            logger.info(f"vLLM-Omni load weights, loaded_params: {len(weights)}")
        else:
            logger.info("Loading standard weights (async)")
            # vLLM v1 GPU workers expose ``reload_weights``; the diffusion
            # worker exposes ``load_weights``. Dispatch by hasattr so this
            # branch handles both worker classes.
            if hasattr(self, "reload_weights"):
                self.reload_weights(weights)
            else:
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
