# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.tokenizer_manager import UpdateWeightsFromTensorReqInput
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.device import get_device_id, get_torch_device, set_expandable_segments
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import convert_weight_keys
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.utils.torch_functional import check_device_is_available
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

from .base import BaseShardingManager

# from vllm.distributed import parallel_state as sglang_ps
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


async def sgl_update_weights(
    engine: Engine,
    params_batch: list[tuple[str, torch.Tensor]],
    device_mesh_key: str,
    device_mesh: DeviceMesh,
    load_format: Optional[str] = None,
):
    """
    Update weights for the inference engine.
    This function is designed to be stateless, so that the caller process could keep the stateful engine.
    Example Use Case:
        - Multiple Producer Process will call this function in a SPMD style

    Args:
        engine: The inference engine created by the caller process.
        params_batch: A list of (name, tensor) tuples. We batched the tensors to avoid the overhead of cpu call.
        device_mesh_key: The key of the device mesh. Typically "tp" or "infer_tp"
        device_mesh: The device mesh.
        load_format: The format of the weights.
    """
    infer_tp_size = device_mesh[device_mesh_key].mesh.size()[0]
    infer_tp_rank = device_mesh[device_mesh_key].get_local_rank()
    from sglang.srt.patch_torch import monkey_patch_torch_reductions

    monkey_patch_torch_reductions()

    # [
    #   (name0, ipc_tensor0_tp0),
    #   (name1, ipc_tensor1_tp0),
    # ]
    named_tensors_batch = [
        (
            name,
            MultiprocessingSerializer.serialize(
                _preprocess_tensor_for_update_weights(tensor)
            ),
        )
        for name, tensor in params_batch
    ]

    if infer_tp_rank == 0:
        gathered_serialized_batches = [None for _ in range(infer_tp_size)]
    else:
        gathered_serialized_batches = None

    # [
    #   [ (name0, ipc_tensor0_tp0), (name1, ipc_tensor1_tp0) ],
    #   [ (name0, ipc_tensor0_tp1), (name1, ipc_tensor1_tp1) ],
    # ]
    dist.gather_object(
        obj=named_tensors_batch,
        object_gather_list=gathered_serialized_batches,
        dst=device_mesh[device_mesh_key].mesh.tolist()[0],
        group=device_mesh[device_mesh_key].get_group(),
    )

    if infer_tp_rank == 0:
        # Use zip(*) to "transpose" the data structure.
        # After transpose, the data structure is like:
        # [
        #   ( (name0, ipc_tensor0_tp0), (name0, ipc_tensor0_tp1) ),
        #   ( (name1, ipc_tensor1_tp0), (name1, ipc_tensor1_tp1) ),
        # ]
        logical_tensors = zip(*gathered_serialized_batches, strict=True)

        named_tensors = [
            # [
            #   (name0, LocalSerializedTensor(values=[ipc_tensor0_tp0, ipc_tensor0_tp1])),
            #   (name1, LocalSerializedTensor(values=[ipc_tensor1_tp0, ipc_tensor1_tp1])),
            # ]
            (
                tensor_group[0][0],
                LocalSerializedTensor(
                    values=[rank_part[1] for rank_part in tensor_group]
                ),
            )
            for tensor_group in logical_tensors
        ]

        update_weights_request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors)
                for _ in range(infer_tp_size)
            ],
            load_format=load_format,
            flush_cache=True
        )

        return await engine.update_weights_from_tensor(update_weights_request)


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    """
    Preprocess the tensor for update weights.
    Example Use Case:
        - FSDP: we gather tensor by calling full_tensor in _preprocess_tensor_for_update_weights
        - Megatron: we do nothing here, assuming it is gathered when feed into this func

    Args:
        tensor: The tensor to be preprocessed.

    Returns:
        The full tensor if it is a DTensor, otherwise the original tensor.
    """
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


class FSDPSGLangShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        rollout_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        multi_stage_wake_up: bool = False,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.rollout_config = rollout_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.multi_stage_wake_up = multi_stage_wake_up

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig()
            )
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.wake_up())

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.sleep())

    async def update_weights(self, params):
        named_tensors = [(k, v) for k, v in params.items()]
        update_weights_bucket_bytes = int(self.rollout_config.update_weights_bucket_megabytes) << 20
        for params_batch in get_named_tensor_buckets(named_tensors, update_weights_bucket_bytes):
            await sgl_update_weights(
                engine=self.inference_engine,
                params_batch=params_batch,
                device_mesh_key="infer_tp",
                device_mesh=self.device_mesh,
            )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            self.inference_engine.flush_cache()

    async def release_memory(self):
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            if self.multi_stage_wake_up:
                await self.inference_engine.release_memory_occupation(tags=["kv_cache", "weights"])
            else:
                await self.inference_engine.release_memory_occupation()
            log_gpu_memory_usage("After release memory occupation in sharding manager", logger=logger)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)
        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        device = get_device_id()  # used when fsdp2 set cpu_offload_policy
        params = {
            k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()
        }

        # convert weight keys to match the model config
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)

        log_gpu_memory_usage("After offload_param in sharding manager memory", logger=logger)

        # sglang need to set _set_allocator_settings to False
        logger.debug("fsdp sglang sharding_manager _set_allocator_settings to False")
        # Note(chenyang): SGLang is using torch memory pool to manage memory
        # which is incompatible with expandable segments
        set_expandable_segments(False)

        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            if self.multi_stage_wake_up:
                await self.inference_engine.resume_memory_occupation(tags=["weights"])
                log_gpu_memory_usage("Before resume SGLang weights in sharding manager", logger=logger)
            else:
                await self.inference_engine.resume_memory_occupation()
                log_gpu_memory_usage("Before resume SGLang weights + kv_cache in sharding manager", logger=logger)

        # Copy, not share memory
        await self.update_weights(params)
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

        del params
        aggressive_empty_cache(force_sync=True)
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        if (
            self.multi_stage_wake_up
            and self.rollout_config.free_cache_engine
            and self.device_mesh["infer_tp"].get_local_rank() == 0
        ):
            await self.inference_engine.resume_memory_occupation(tags=["kv_cache"])
            log_gpu_memory_usage("After resume SGLang kv_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        if self.rollout_config.free_cache_engine:
            log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
            await self.release_memory()
            log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)

        # always set _set_allocator_settings to True when using sglang
        # it is required by fsdp2 to avoid oom
        logger.debug("fsdp sglang sharding_manager _set_allocator_settings to True")
        set_expandable_segments(True)

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = self.device_mesh["infer_tp"].get_group()

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
