import inspect
import os
from typing import Optional, Dict, Any
import torch
import torch.distributed as dist
from torch.distributed import all_gather as all_gather
from torch.distributed.nn.functional import all_gather as all_gather_grads

import logging
logger = logging.getLogger(__name__)

_CPU_GATHER_GROUP = None
_INIT_PG_SUPPORTS_DEVICE_ID = "device_id" in inspect.signature(dist.init_process_group).parameters
_BARRIER_SUPPORTS_DEVICE_IDS = "device_ids" in inspect.signature(dist.barrier).parameters

def init_distributed() -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = 0
    local_rank = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_kwargs = {}
        if torch.cuda.is_available() and _INIT_PG_SUPPORTS_DEVICE_ID:
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend=backend, **init_kwargs)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
        logger.info(f"Initialized distributed inference rank {rank}/{world_size} (local_rank={local_rank}) on device {device}")
    else:
        logger.info(f"Running single-process inference on device {device}")

    return distributed, world_size, rank, local_rank, device

def finalize_distributed(distributed: bool, local_rank: int = 0):
    if distributed and dist.is_initialized():
        barrier_kwargs: Dict[str, Any] = {}
        if torch.cuda.is_available() and _BARRIER_SUPPORTS_DEVICE_IDS:
            barrier_kwargs["device_ids"] = [local_rank]
        dist.barrier(**barrier_kwargs)
        dist.destroy_process_group()

def get_gloo_group():
    """
    When NCCL is the backend, all_gather_object allocates CUDA byte buffers,
    which can OOM if the payload is large. Create a CPU (gloo) group for
    object collection to keep the buffers off GPU.
    """
    gather_group = None
    if dist.is_initialized():
        try:
            if dist.get_backend() != "gloo":
                global _CPU_GATHER_GROUP
                if _CPU_GATHER_GROUP is None:
                    _CPU_GATHER_GROUP = dist.new_group(backend="gloo")
                gather_group = _CPU_GATHER_GROUP
        except Exception:
            logger.exception("Falling back to default process group for all_gather_object")
    return gather_group

def dist_gather_tensor(t: Optional[torch.Tensor], world_size: int, process_rank: int):
    if t is None:
        return None
    t = t.contiguous()

    if t.dim() > 1:
        local_shape = torch.tensor(t.shape[1:], device=t.device, dtype=torch.long)
        shape_list = [local_shape.clone() for _ in range(world_size)]
        all_gather(shape_list, local_shape)
        max_shape = torch.stack(shape_list).max(dim=0).values.tolist() if shape_list else []
        if any(target > current for target, current in zip(max_shape, t.shape[1:])):
            padded_shape = (t.shape[0],) + tuple(max_shape)
            padded = t.new_zeros(padded_shape)
            slices = [slice(None)]
            for dim_len in t.shape[1:]:
                slices.append(slice(0, dim_len))
            padded[tuple(slices)] = t
            t = padded

    all_tensors = [torch.empty_like(t) for _ in range(world_size)]
    all_gather(all_tensors, t)

    all_tensors[process_rank] = t
    all_tensors = torch.cat(all_tensors, dim=0)

    return all_tensors


def dist_gather_tensor_with_grads(t: Optional[torch.Tensor], world_size: int, process_rank: int):
    if t is None:
        return None
    t = t.contiguous()

    if t.dim() > 1:
        local_shape = torch.tensor(t.shape[1:], device=t.device, dtype=torch.long)
        shape_list = [local_shape.clone() for _ in range(world_size)]
        all_gather(shape_list, local_shape)
        max_shape = torch.stack(shape_list).max(dim=0).values.tolist() if shape_list else []
        if any(target > current for target, current in zip(max_shape, t.shape[1:])):
            padded_shape = (t.shape[0],) + tuple(max_shape)
            padded = t.new_zeros(padded_shape)
            slices = [slice(None)]
            for dim_len in t.shape[1:]:
                slices.append(slice(0, dim_len))
            padded[tuple(slices)] = t
            t = padded

    tensor_list = all_gather_grads(t)
    all_tensors = torch.cat(tensor_list, dim=0)

    return all_tensors

def list_array_split_np_style(data, n):
    length = len(data)
    size = length // n
    remainder = length % n
    
    result = []
    start = 0
    for i in range(n):
        # Add 1 to the size if we still have remainder to distribute
        end = start + size + (1 if i < remainder else 0)
        result.append(data[start:end])
        start = end
        
    return result