from typing import Optional, List
import torch
import numpy as np
from contextlib import nullcontext

import logging
logger = logging.getLogger(__name__)

def colbert_sim(
    query: torch.Tensor,    # [B,Q,D]
    doc: torch.Tensor,      # [B,S,D]
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
    normalize: bool = False,
    return_argmax: bool = False,
    use_full_precision: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if use_full_precision:
        autocast_context = torch.amp.autocast(device_type=query.device.type, enabled=False)
    else:
        autocast_context = nullcontext()
    with autocast_context:
        if use_full_precision:
            query = query.float()
            doc = doc.float()
        else:
            doc = doc.to(dtype=query.dtype)

        sim = torch.einsum("bqd,csd->bcqs", query, doc) # [query_batch_size, doc_batch_size, len_query, len_doc]

        # mask out the padding tokens in the document to -inf
        if doc_mask is not None:
            doc_mask = doc_mask.to(device=sim.device, dtype=torch.bool) # shape: [doc_batch_size, len_doc]
            sim = sim.masked_fill(~doc_mask.unsqueeze(0).unsqueeze(2), float("-inf"))

        if return_argmax:
            sim_argmax = sim.argmax(dim=-1)

        # max-sum
        sim = sim.amax(dim=-1)
        # mask out the padding tokens in the query to 0s
        if query_mask is not None:
            query_mask = query_mask.to(device=sim.device, dtype=torch.bool) # shape: [query_batch_size, len_query]
            sim = sim.masked_fill(~query_mask.unsqueeze(1), 0)
        sim = sim.sum(dim=-1)

        # query length normalization
        if normalize and query_mask is not None:
            sim = sim / torch.clamp(query_mask.sum(dim=1,keepdim=True), min=1.0)

        if return_argmax:
            return sim, sim_argmax
        return sim

def pad_and_concat(arrays: List[np.ndarray], is_equal_len: Optional[bool] = None):
    if arrays is None or arrays[0] is None:
        return None

    if is_equal_len is None:
        first_len = arrays[0].shape[1]
        is_equal_len = all(arr.shape[1] == first_len for arr in arrays)

    if is_equal_len:
        return np.concatenate(arrays, axis=0)
    else:
        max_len = max(arr.shape[1] for arr in arrays)
        padded = []
        for arr in arrays:
            padding_len = max_len - arr.shape[1]
            if padding_len > 0:
                pad_shape = (arr.shape[0], padding_len, *arr.shape[2:])
                padding = np.zeros(pad_shape, dtype=arr.dtype)
                arr = np.concatenate([arr, padding], axis=1)
            padded.append(arr)
        return np.concatenate(padded, axis=0)

def pad_and_stack(arrays: List[np.ndarray], is_equal_len: Optional[bool] = None):
    if arrays is None or arrays[0] is None:
        return None

    if is_equal_len is None:
        first_len = arrays[0].shape[0]
        is_equal_len = all(arr.shape[0] == first_len for arr in arrays)

    if is_equal_len:
        return np.stack(arrays, axis=0)
    else:
        max_len = max(arr.shape[0] for arr in arrays)
        padded = []
        for arr in arrays:
            padding_len = max_len - arr.shape[0]
            if padding_len > 0:
                pad_shape = (padding_len, *arr.shape[1:])
                padding = np.zeros(pad_shape, dtype=arr.dtype)
                arr = np.concatenate([arr, padding], axis=0)
            padded.append(arr)
        return np.stack(padded, axis=0)