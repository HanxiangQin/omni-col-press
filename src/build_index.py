#!/usr/bin/env python3

import os
import sys
import logging
import json
import numpy as np
import torch
import torch.distributed as dist
import torch.profiler as profiler
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from contextlib import nullcontext
from dataclasses import asdict

from src.arguments import DataArguments, ModelArguments, IndexArguments
from src.collator import BaseMultiModalCollator
from src.index import MultiModalIndex
from src.factory.components_factory import build_index
from src.factory.factory import create_inference_components
from src.utils import init_distributed, finalize_distributed, get_gloo_group, pad_and_concat
from transformers.hf_argparser import HfArgumentParser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args() -> Tuple[ModelArguments, DataArguments, IndexArguments]:
    parser = HfArgumentParser((ModelArguments, DataArguments, IndexArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, index_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        model_args, data_args, index_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, index_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        index_args: IndexArguments

    return model_args, data_args, index_args

def _get_build_config(model_args: ModelArguments, data_args: DataArguments, index_args: IndexArguments, num_ids: int, index_shape):
    return {
        'model_name_or_path': model_args.model_name_or_path,
        'lora_name_or_path': model_args.lora_name_or_path,
        'corpus_path': data_args.corpus_path,
        'index_type': index_args.index_type,
        'modality_config': data_args.resolve_modalities(is_query=False),
        'append_eos_token': data_args.append_eos_token,
        'num_appending_token': model_args.num_appending_token,
        'num_repr_vectors': model_args.num_repr_vectors,
        'pooling': model_args.pooling,
        'normalize': model_args.normalize,
        'passage_prefix': data_args.passage_prefix,
        'num_ids': num_ids,
        'index_shape': index_shape,
        'batch_size': index_args.batch_size,
        'model_args': asdict(model_args),
        'data_args': asdict(data_args),
        'index_args': asdict(index_args)
    }

def _save_failures(failures: List[Dict[str, Any]], index_output_path: str):
    try:
        if failures and isinstance(failures, list) and len(failures) > 0:
            os.makedirs(index_output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fail_log = os.path.join(index_output_path, f"failed_media_{timestamp}.jsonl")
            with open(fail_log, 'w') as f:
                for rec in failures:
                    rec_out = {"timestamp": datetime.now().isoformat(), **rec}
                    f.write(json.dumps(rec_out) + "\n")
            logger.info(f"Wrote {len(failures)} failure records to {fail_log}")
    except Exception:
        logger.exception("Failed to persist failure records")

def _save(index_output_path: str, index: MultiModalIndex, failures: Optional[List[Dict[str, Any]]], output_pickle: bool = False, build_config: Dict = {}):
    os.makedirs(index_output_path, exist_ok=True)

    logger.info(f"Saving index to {index_output_path}...")
    index.save(index_output_path)

    if output_pickle:
        pickle_path = os.path.join(index_output_path, f"index.pkl")
        logger.info(f"Saving index to pickle file: {pickle_path}")
        index.save_pickle(pickle_path)
    
    config_path = os.path.join(index_output_path, f"build_config.json")
    with open(config_path, 'w') as f:
        json.dump(build_config, f)

    _save_failures(failures, index_output_path)

def _encode_data(model, dataloader, device, is_var_len: bool, quiet=False, max_batches=None, enable_profiling=False, profile_batches=10):
    all_embeddings = []
    all_ids = []
    all_metadata = []
    skipped_batches = 0
    if is_var_len:
        all_masks = []
    else:
        all_masks = None
    
    # Check if profiling is enabled via environment variable
    enable_profiling = enable_profiling or os.getenv("ENABLE_PROFILING", "0") == "1"
    profile_output_dir = os.getenv("PROFILE_OUTPUT_DIR", "./log/profiler")
    
    model.eval()
    
    # Setup profiler if enabled
    if enable_profiling:
        logger.info(f"Profiling enabled: will profile {profile_batches} batches")
        logger.info(f"Profile output directory: {profile_output_dir}")
        os.makedirs(profile_output_dir, exist_ok=True)
        
        prof_context = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(
                wait=1,      # Skip first batch
                warmup=1,    # Warmup for 1 batch
                active=profile_batches,  # Profile active batches
                repeat=1
            ),
            on_trace_ready=profiler.tensorboard_trace_handler(profile_output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        prof_context = nullcontext()
    
    with torch.inference_mode():
        with prof_context as prof:
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding corpus", disable=quiet)):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                ids = inputs = None
                if "ids" in batch and "inputs" in batch:
                    ids = batch["ids"]
                    inputs = batch["inputs"]
                elif "passage_ids" in batch and "passage_inputs" in batch:
                    ids = batch["passage_ids"]
                    inputs = batch["passage_inputs"]
                if ids is None or inputs is None or len(ids) == 0:
                    logger.warning(f"Empty batch at idx={batch_idx}; skipping")
                    skipped_batches += 1
                    continue

                for k in inputs:
                    inputs[k] = inputs[k].to(device)
                
                embeddings, masks = model.encode_passage(inputs)
                embeddings = embeddings.cpu().detach().numpy()
                
                all_embeddings.append(embeddings)
                all_ids.extend(ids)
                all_metadata.extend([{"id": id} for id in ids])
                if all_masks is not None:
                    masks = masks.cpu().detach().numpy()
                    all_masks.append(masks)
                
                # Step profiler if enabled
                if enable_profiling and prof is not None:
                    prof.step()
    
    # Print profiling results if enabled
    if enable_profiling and prof is not None:
        logger.info("=" * 80)
        logger.info("PROFILING RESULTS - Top operations by CUDA time:")
        logger.info("=" * 80)
        try:
            key_averages = prof.key_averages()
            logger.info(key_averages.table(
                sort_by="cuda_time_total",
                row_limit=30,
                max_name_column_width=100
            ))
            
            # Also print by CPU time
            logger.info("\n" + "=" * 80)
            logger.info("PROFILING RESULTS - Top operations by CPU time:")
            logger.info("=" * 80)
            logger.info(key_averages.table(
                sort_by="cpu_time_total",
                row_limit=30,
                max_name_column_width=100
            ))
            
            # Export Chrome trace
            trace_file = os.path.join(profile_output_dir, "trace.json")
            prof.export_chrome_trace(trace_file)
            logger.info(f"\nChrome trace exported to: {trace_file}")
            logger.info("View it in Chrome by navigating to chrome://tracing and loading the trace file")
            
        except Exception as e:
            logger.warning(f"Failed to process profiling results: {e}")
    
    return all_embeddings, all_masks, all_ids, all_metadata

def _gather_encoded_results(
    embedding_list: Optional[np.ndarray],
    mask_list: Optional[np.ndarray],
    ids: List[str],
    metadata: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    rank: int,
    world_size: int,
    is_var_len: bool,
) -> (List[np.ndarray], List[str], List[Dict[str, Any]], List[Dict[str, Any]]):
    if world_size == 1:
        merged_embedding_list = embedding_list
        merged_ids = ids
        merged_metadata = metadata
        merged_failures = failures
        if is_var_len:
            merged_mask_list = mask_list

    else:
        payload = {
        "embedding_list": embedding_list,
        "ids": ids,
        "metadata": metadata,
        "failures": failures,
        }
        if is_var_len:
            payload["mask_list"] = mask_list
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]

        gather_group = get_gloo_group()
        dist.all_gather_object(gathered, payload, group=gather_group)

        if rank != 0:
            return None, None, [], [], []

        merged_embedding_list: List[np.ndarray] = []
        merged_ids: List[str] = []
        merged_metadata: List[Dict[str, Any]] = []
        merged_failures: List[Dict[str, Any]] = []
        if is_var_len:
            merged_mask_list: List[np.ndarray] = []
        else:
            merged_mask_list = None

        for entry in gathered:
            if entry is None:
                continue
            merged_embedding_list.extend(entry.get("embedding_list", []))
            merged_ids.extend(entry.get("ids", []))
            merged_metadata.extend(entry.get("metadata", []))
            merged_failures.extend(entry.get("failures", []))
            if is_var_len:
                merged_mask_list.extend(entry.get("mask_list", []))

        if not merged_embedding_list:
            logger.warning("Main rank received no embeddings from workers.")
            return None, None, merged_ids, merged_metadata, merged_failures
        
    merged_embeddings = pad_and_concat(arrays=merged_embedding_list, is_equal_len=not is_var_len)
    merged_masks = pad_and_concat(arrays=merged_mask_list, is_equal_len=False) if is_var_len else None
    
    return merged_embeddings, merged_masks, merged_ids, merged_metadata, merged_failures

def encode_and_gather(
    model,
    dataloader,
    torch_dtype,
    device_for_inputs,
    collator: BaseMultiModalCollator,
    rank: int,
    world_size: int,
    is_var_len: bool,
    quiet: bool = False,
    enable_profiling: bool = False,
    profile_batches: int = 10,
) -> Tuple[Optional[np.ndarray], List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    use_autocast = (
        torch_dtype in [torch.float16, torch.bfloat16]
        and isinstance(device_for_inputs, str)
        and device_for_inputs.startswith("cuda")
    )
    autocast_context = (
        torch.amp.autocast(device_type='cuda', dtype=torch_dtype) if use_autocast else nullcontext()
    )
    with autocast_context:
        embeddings_list, masks_list, ids, metadata = _encode_data(
            model=model,
            dataloader=dataloader,
            device=device_for_inputs,
            is_var_len=is_var_len,
            quiet=quiet,
            enable_profiling=enable_profiling and (rank == 0),  # Only profile on main process
            profile_batches=profile_batches,
        )
    failures = collator.get_failures()
    embeddings, masks, ids, metadata, failures = _gather_encoded_results(
        embeddings_list,
        masks_list,
        ids,
        metadata,
        failures,
        rank,
        world_size,
        is_var_len,
    )
    return embeddings, masks, ids, metadata, failures

def main():
    distributed, world_size, rank, local_rank, device = init_distributed()
    is_main_process = rank == 0
    model_args, data_args, index_args = parse_args()

    if is_main_process:
        logger.info("Initializing model/processor/collator via factory...")
    model, processor, collator, dataset, dataloader, device_for_inputs, torch_dtype = create_inference_components(
        device=device,
        model_args=model_args,
        data_args=data_args,
        batch_size=index_args.batch_size,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    # Check if profiling is enabled via environment variable
    enable_profiling = os.getenv("ENABLE_PROFILING", "0") == "1"
    profile_batches = int(os.getenv("PROFILE_BATCHES", "10"))
    
    embeddings, masks, doc_ids, metadata, failures = encode_and_gather(
        model=model,
        dataloader=dataloader,
        torch_dtype=torch_dtype,
        device_for_inputs=device_for_inputs,
        collator=collator,
        rank=rank,
        world_size=world_size,
        is_var_len=(model.pooling in ["colbert", "select"]),
        quiet=not is_main_process,
        enable_profiling=enable_profiling,
        profile_batches=profile_batches,
    )

    if not is_main_process:
        finalize_distributed(distributed, local_rank)
        return

    del model
    logger.info(f"Building type:{index_args.index_type} index ...")
    build_config = _get_build_config(model_args, data_args, index_args, len(doc_ids), embeddings.shape)
    index = build_index(index_args.index_type, embeddings, masks, doc_ids, metadata, device, build_config, index_args.index_output_path)
    
    _save(index_args.index_output_path, index, failures, index_args.output_pickle, build_config)
    
    logger.info(f"Index built successfully with {len(doc_ids)} documents across {world_size} process(es)!")
    logger.info(f"Statistics: {index.get_stats()}")

    finalize_distributed(distributed, local_rank)

if __name__ == "__main__":
    main()