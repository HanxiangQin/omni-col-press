#!/usr/bin/env python3

import pickle
import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import HfArgumentParser
from contextlib import nullcontext

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.arguments import DataArguments, ModelArguments, EvaluateArguments
from src.index import MultiModalIndex
from src.factory.factory import create_inference_components
from src.utils import (
    init_distributed,
    finalize_distributed,
    load_index,
    load_ground_truth,
    pad_and_concat,
    get_relevant_docs_list,
    calculate_recall_at_k,
    calculate_ndcg_at_k,
    calculate_mrr,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    model_args: ModelArguments
    data_args: DataArguments
    eval_args: EvaluateArguments
    parser = HfArgumentParser((ModelArguments, DataArguments, EvaluateArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, eval_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        model_args, data_args, eval_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, eval_args

def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{s}\n')

def _eval(
    model,
    dataloader,
    device_for_inputs,
    torch_dtype,
    index: MultiModalIndex,
    ground_truth: Dict[str, Dict[str, int]],
    top_k_values: List[int],
    quiet: bool = False,
    collect_query_embeddings: bool = False,
    collect_rankings: bool = False,
):
    max_k = max(top_k_values)
    recall_sums = {k: 0.0 for k in top_k_values}
    ndcg_sums = {k: 0.0 for k in top_k_values}
    mrr_sum = 0.0
    valid_queries = 0

    collect_lookup_indices = collect_query_embeddings or collect_rankings
    query_embeddings: List[np.ndarray] = []
    lookup_indices: List[str] = []
    all_retrieved_doc_ids: List[List[str]] = []
    all_scores: List[Any] = []

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating", disable=quiet):
            query_ids = batch["ids"]
            inputs = batch["inputs"]

            for input_name in inputs:
                inputs[input_name] = inputs[input_name].to(device_for_inputs)

            outputs, query_mask = model.encode_query(inputs)
            embeddings = outputs.cpu().detach().numpy()

            if collect_query_embeddings:
                query_embeddings.append(embeddings)

            scores, indices = index.batch_search(outputs, k=max_k, quiet=quiet, query_mask=query_mask)
            retrieved_doc_ids = index.get_doc_ids(indices)

            if collect_lookup_indices:
                lookup_indices.extend(query_ids)
            if collect_rankings:
                all_retrieved_doc_ids.extend(retrieved_doc_ids)
                all_scores.extend(scores)

            for query_id, retrieved_docs in zip(query_ids, retrieved_doc_ids):
                relevant_docs_list, relevant_docs_dict = get_relevant_docs_list(query_id, ground_truth)
                if not relevant_docs_list:
                    continue

                valid_queries += 1
                for k in top_k_values:
                    recall_sums[k] += calculate_recall_at_k(retrieved_docs, relevant_docs_list, k)
                    ndcg_sums[k] += calculate_ndcg_at_k(retrieved_docs, relevant_docs_dict, k)
                mrr_sum += calculate_mrr(retrieved_docs, relevant_docs_list)

    return {
        "recall_sums": recall_sums,
        "ndcg_sums": ndcg_sums,
        "mrr_sum": mrr_sum,
        "valid_queries": valid_queries,
        "query_embeddings": query_embeddings,
        "lookup_indices": lookup_indices,
        "retrieved_doc_ids": all_retrieved_doc_ids,
        "scores": all_scores,
    }


def _gather_eval_results(
    eval_payload: Dict[str, Any],
    top_k_values: List[int],
    rank: int,
    world_size: int,
):
    if world_size == 1:
        return eval_payload

    gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, eval_payload)

    if rank != 0:
        return None

    merged = {
        "recall_sums": {k: 0.0 for k in top_k_values},
        "ndcg_sums": {k: 0.0 for k in top_k_values},
        "mrr_sum": 0.0,
        "valid_queries": 0,
        "query_embeddings": [],
        "lookup_indices": [],
        "retrieved_doc_ids": [],
        "scores": [],
    }

    for payload in gathered:
        if payload is None:
            continue
        for k in top_k_values:
            merged["recall_sums"][k] += payload["recall_sums"].get(k, 0.0)
            merged["ndcg_sums"][k] += payload["ndcg_sums"].get(k, 0.0)
        merged["mrr_sum"] += payload.get("mrr_sum", 0.0)
        merged["valid_queries"] += payload.get("valid_queries", 0)
        merged["query_embeddings"].extend(payload.get("query_embeddings", []))
        merged["lookup_indices"].extend(payload.get("lookup_indices", []))
        merged["retrieved_doc_ids"].extend(payload.get("retrieved_doc_ids", []))
        merged["scores"].extend(payload.get("scores", []))

    return merged


def _compute_metrics(aggregated_payload: Dict[str, Any], top_k_values: List[int]) -> Dict[str, Any]:
    valid_queries = aggregated_payload.get("valid_queries", 0)
    if valid_queries == 0:
        logger.warning("No valid queries found for evaluation!")
        return {"num_queries": 0}

    results = {}
    for k in top_k_values:
        recall = aggregated_payload["recall_sums"][k] / valid_queries
        ndcg = aggregated_payload["ndcg_sums"][k] / valid_queries
        results[f"Recall@{k}"] = recall
        results[f"NDCG@{k}"] = ndcg
    results["MRR"] = aggregated_payload["mrr_sum"] / valid_queries
    results["num_queries"] = valid_queries
    return results


def eval_and_gather(
    model,
    dataloader,
    device_for_inputs,
    torch_dtype,
    index: MultiModalIndex,
    ground_truth: Dict[str, Dict[str, int]],
    top_k_values: List[int],
    rank: int,
    world_size: int,
    collect_query_embeddings: bool = False,
    collect_rankings: bool = False,
    quiet: bool = False,
):
    use_autocast = (
        torch_dtype in [torch.float16, torch.bfloat16]
        and isinstance(device_for_inputs, str)
        and device_for_inputs.startswith("cuda")
    )
    autocast_context = (
        torch.amp.autocast(device_type='cuda', dtype=torch_dtype) if use_autocast else nullcontext()
    )
    with autocast_context:
        eval_payload = _eval(
            model=model,
            dataloader=dataloader,
            device_for_inputs=device_for_inputs,
            torch_dtype=torch_dtype,
            index=index,
            ground_truth=ground_truth,
            top_k_values=top_k_values,
            quiet=quiet,
            collect_query_embeddings=collect_query_embeddings,
            collect_rankings=collect_rankings,
        )

    aggregated_payload = _gather_eval_results(
        eval_payload=eval_payload,
        top_k_values=top_k_values,
        rank=rank,
        world_size=world_size,
    )

    if aggregated_payload is None:
        return None, None

    results = _compute_metrics(aggregated_payload, top_k_values)
    return results, aggregated_payload

def main():
    distributed, world_size, rank, local_rank, device = init_distributed()
    is_main_process = rank == 0
    model_args, data_args, eval_args = parse_args()

    if is_main_process:
        logger.info(f"eval_args: {eval_args} \ndata_args: {data_args} \nmodel_args: {model_args}")
        logger.info("Initializing model/processor/collator via factory...")
    model, _, _, query_dataset, dataloader, device_for_inputs, torch_dtype = create_inference_components(
        device=device,
        model_args=model_args,
        data_args=data_args,
        batch_size=eval_args.batch_size,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    index: MultiModalIndex = load_index(eval_args, device)
    ground_truth = load_ground_truth(data_args, query_dataset)
    top_k_values = eval_args.top_k

    collect_query_embeddings = bool(eval_args.output_query_embeddings)
    collect_rankings = bool(eval_args.output_rank_file)
    quiet = not is_main_process

    results, aggregated_payload = eval_and_gather(
        model=model,
        dataloader=dataloader,
        device_for_inputs=device_for_inputs,
        torch_dtype=torch_dtype,
        index=index,
        ground_truth=ground_truth,
        top_k_values=top_k_values,
        rank=rank,
        world_size=world_size,
        collect_query_embeddings=collect_query_embeddings,
        collect_rankings=collect_rankings,
        quiet=quiet,
    )

    if not is_main_process:
        finalize_distributed(distributed, local_rank)
        return

    if results is None or aggregated_payload is None:
        logger.error("Failed to gather evaluation results on the main process.")
        finalize_distributed(distributed, local_rank)
        return

    logger.info("Evaluation Results:" + "\n" + "=" * 50)
    for metric, score in results.items():
        if metric != "num_queries":
            logger.info(f"{metric}: {score:.4f}")
    logger.info(f"Number of evaluated queries: {results['num_queries']}")

    if eval_args.output_path:
        os.makedirs(eval_args.output_path, exist_ok=True)
        logger.info(f"Saving results to {eval_args.output_path}...")

        index_stats = index.get_stats()
        output_data = {
            "evaluation_results": results,
            "configuration": {
                "index_path": eval_args.index_path,
                "query_path": data_args.query_path,
                "model_name_or_path": model_args.model_name_or_path,
                "top_k": top_k_values,
                "query_modality_config": data_args.resolve_modalities(is_query=True),
            },
            "index_stats": index_stats,
        }

        result_file = os.path.join(eval_args.output_path, "results.json")
        with open(result_file, "w") as f:
            json.dump(output_data, f, indent=2)

        if collect_query_embeddings and aggregated_payload["query_embeddings"]:
            is_var_len = model_args.pooling in ['colbert', 'select']
            query_embeddings = pad_and_concat(arrays=aggregated_payload["query_embeddings"], is_equal_len=not is_var_len)
            query_embeddings_file = os.path.join(eval_args.output_path, "query_embeddings.pkl")
            with open(query_embeddings_file, "wb") as f:
                pickle.dump((query_embeddings, aggregated_payload["lookup_indices"]), f)

        if collect_rankings and aggregated_payload["retrieved_doc_ids"]:
            ranking_file_path = os.path.join(eval_args.output_path, "ranking.txt")
            write_ranking(
                aggregated_payload["retrieved_doc_ids"],
                aggregated_payload["scores"],
                aggregated_payload["lookup_indices"],
                ranking_file_path,
            )

    stats = index.get_stats()
    logger.info(f"Index statistics: {stats}")

    finalize_distributed(distributed, local_rank)


if __name__ == "__main__":
    main()