import os
import gc
import tempfile
import shutil
import time
from contextlib import contextmanager
from typing import Optional

import torch
import numpy as np

from transformers.trainer import Trainer, TRAINING_ARGS_NAME
from transformers.trainer_utils import speed_metrics
import torch.distributed as dist
from src.encoder import EncoderModel
from src.utils import (
    pad_and_stack,
    get_relevant_docs_list,
    calculate_recall_at_k,
    calculate_ndcg_at_k,
    calculate_mrr,
    dist_gather_tensor,
    get_gloo_group,
    list_array_split_np_style,
    load_dataset_labels,
)
from src.arguments import IndexArguments
from src.factory.components_factory import build_index

import logging
logger = logging.getLogger(__name__)


class RetrievalTrainer(Trainer):
    def __init__(self, index_args: IndexArguments, *args, **kwargs):
        super(RetrievalTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_ddp else 0
        self.world_size = dist.get_world_size() if self.is_ddp else 1
        self._dist_loss_scale_factor = self.world_size if self.is_ddp else 1
        self.can_return_loss = True
        self.index_args = index_args

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")

        # save model weights (delegated to EncoderModel.save)
        # this handles splitting base transformer weights and extra encoder modules
        self.model.save(output_dir, state_dict_override=state_dict)

        # save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # save training arguments
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    @contextmanager
    def _enable_eval_loss(self):
        encoder = self.model
        if not hasattr(encoder, "return_loss"):
            yield
            return
        previous = encoder.return_loss
        encoder.return_loss = True
        try:
            yield
        finally:
            encoder.return_loss = previous

    def _merge_distributed_payloads(self, loss_sum, num_batches, all_query_embeddings, all_passage_embeddings):
        """Gather and merge encoding results from all processes."""
        payload = {
            "loss_sum": loss_sum,
            "num_batches": num_batches,
            "all_query_embeddings": all_query_embeddings,
            "all_passage_embeddings": all_passage_embeddings,
        }
        if self.is_ddp:
            payload_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(payload_list, payload, group=get_gloo_group())
        else:
            payload_list = [payload]

        # Merge payloads from all processes
        total_loss_sum = 0.0
        total_num_batches = 0
        merged_query_embeddings = {}
        merged_passage_embeddings = {}
        for payload_item in payload_list:
            total_loss_sum += payload_item["loss_sum"]
            total_num_batches += payload_item["num_batches"]
            merged_query_embeddings.update(payload_item["all_query_embeddings"])
            merged_passage_embeddings.update(payload_item["all_passage_embeddings"])
        
        del payload_list
        gc.collect()

        loss = total_loss_sum / total_num_batches if total_num_batches > 0 else 0.0
        return loss, merged_query_embeddings, merged_passage_embeddings

    def _embeddings_dict_to_arrays(self, embeddings_dict, is_var_len):
        """Convert embeddings dictionary to sorted lists and stacked arrays.
        
        Args:
            embeddings_dict: Dict mapping id -> (embedding, mask)
            is_var_len: Whether embeddings have variable length
            
        Returns:
            Tuple of (ids_list, embeddings_array, masks_array)
        """
        ids_list = []
        embeddings_list = []
        masks_list = []
        # Sort IDs to ensure deterministic order regardless of dictionary merge order
        for item_id in sorted(embeddings_dict.keys()):
            embedding, mask = embeddings_dict[item_id]
            ids_list.append(item_id)
            embeddings_list.append(embedding)
            masks_list.append(mask)
        
        embeddings = pad_and_stack(arrays=embeddings_list, is_equal_len=not is_var_len)
        masks = pad_and_stack(arrays=masks_list, is_equal_len=not is_var_len)
        
        del embeddings_list, masks_list
        gc.collect()
        
        return ids_list, embeddings, masks

    def _gather_encoding_results(self, loss_sum, num_batches, all_query_embeddings, all_passage_embeddings):
        """Gather encoding results from all processes and prepare for retrieval.
        
        Returns:
            Tuple of (loss, local_query_ids, local_query_embeddings, local_query_masks,
                     passage_ids_list, passage_embeddings, passage_masks)
        """
        # Merge results from all processes
        loss, merged_query_embeddings, merged_passage_embeddings = self._merge_distributed_payloads(
            loss_sum, num_batches, all_query_embeddings, all_passage_embeddings
        )
        
        # Convert to arrays (queries always have variable length)
        is_var_len = self.model.pooling in ['colbert', 'select']
        query_ids_list, query_embeddings, query_masks = self._embeddings_dict_to_arrays(
            merged_query_embeddings, is_var_len=True
        )
        passage_ids_list, passage_embeddings, passage_masks = self._embeddings_dict_to_arrays(
            merged_passage_embeddings, is_var_len=is_var_len
        )
        
        del merged_query_embeddings, merged_passage_embeddings
        gc.collect()

        # Split queries across processes for parallel retrieval
        local_query_embeddings = np.array_split(query_embeddings, self.world_size)[self.rank]
        local_query_masks = np.array_split(query_masks, self.world_size)[self.rank]
        local_query_ids = list_array_split_np_style(query_ids_list, self.world_size)[self.rank]

        return loss, local_query_ids, local_query_embeddings, local_query_masks, passage_ids_list, passage_embeddings, passage_masks

    def _eval_loop(self, inputs):
        """Run a single evaluation step."""
        inputs = self._prepare_inputs(inputs)
        query_inputs = inputs["query_inputs"]
        passage_inputs = inputs["passage_inputs"]
        passage_ids_tensor = inputs["passage_ids_tensor"]
        outputs = self.model(
            query_inputs=query_inputs,
            passage_inputs=passage_inputs,
            passage_ids=passage_ids_tensor,
        )
        return outputs

    def _extract_batch_outputs(self, outputs, query_ids, passage_ids, all_query_embeddings, all_passage_embeddings):
        """Extract and store embeddings from model outputs.
        
        Args:
            outputs: Model outputs containing q_reps, q_mask, p_reps, p_mask
            query_ids: List of query IDs for this batch
            passage_ids: List of passage IDs for this batch
            all_query_embeddings: Dict to store query embeddings (modified in-place)
            all_passage_embeddings: Dict to store passage embeddings (modified in-place)
        """
        # Store query embeddings
        for query_id, query_embedding, query_mask in zip(
            query_ids,
            outputs.q_reps.detach().cpu().float().numpy(),
            outputs.q_mask.detach().cpu().numpy(),
        ):
            all_query_embeddings[query_id] = (query_embedding, query_mask)
        
        # Store passage embeddings (mask could be None if fixed-length)
        if outputs.p_mask is not None:
            for passage_id, passage_embedding, passage_mask in zip(
                passage_ids,
                outputs.p_reps.detach().cpu().float().numpy(),
                outputs.p_mask.detach().cpu().numpy(),
            ):
                all_passage_embeddings[passage_id] = (passage_embedding, passage_mask)
        else:
            for passage_id, passage_embedding in zip(
                passage_ids,
                outputs.p_reps.detach().cpu().float().numpy(),
            ):
                all_passage_embeddings[passage_id] = (passage_embedding, None)

    def _encode_all(self, eval_dataset):
        """Encode all queries and passages in the evaluation dataset.
        
        Returns:
            Tuple of (loss_sum, num_batches, all_query_embeddings, all_passage_embeddings)
        """
        dataloader = self.get_eval_dataloader(eval_dataset)
        self.callback_handler.eval_dataloader = dataloader
        
        num_batches = 0
        loss_sum = 0.0
        all_query_embeddings = {}
        all_passage_embeddings = {}

        with self._enable_eval_loss():
            with torch.no_grad():
                for inputs in dataloader:
                    num_batches += 1
                    query_ids = inputs["query_ids"]
                    passage_ids = inputs["passage_ids"]
                    
                    outputs = self._eval_loop(inputs)
                    loss_sum += outputs.loss.item()
                    
                    self._extract_batch_outputs(
                        outputs, query_ids, passage_ids,
                        all_query_embeddings, all_passage_embeddings
                    )
                    
                    self.control = self.callback_handler.on_prediction_step(
                        self.args, self.state, self.control
                    )
        
        return loss_sum, num_batches, all_query_embeddings, all_passage_embeddings

    def _build_index_and_retrieve(self, query_embeddings, query_masks, passage_embeddings, passage_masks, passage_ids_list, top_k):
        """Build index and retrieve top-k documents for each query.
        
        Args:
            query_embeddings: Query embedding array
            query_masks: Query mask array
            passage_embeddings: Passage embedding array
            passage_masks: Passage mask array
            passage_ids_list: List of passage IDs
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (retrieved_doc_ids, index_time, retrieval_time) or (None, 0, 0) if no queries
        """
        if len(query_embeddings) == 0:
            return None, 0.0, 0.0
        
        temp_dir = None
        index = None
        try:
            temp_dir = tempfile.mkdtemp()
            
            index_start = time.time()
            index = build_index(
                self.index_args.index_type,
                passage_embeddings, passage_masks, passage_ids_list,
                None, self.args.device, None, temp_dir
            )
            index_time = time.time() - index_start
            
            retrieval_start = time.time()
            scores, indices = index.batch_search(
                query_embeddings, query_masks, k=top_k, batch_size=10, quiet=True
            )
            retrieved_doc_ids = index.get_doc_ids(indices)
            retrieval_time = time.time() - retrieval_start
            
            return retrieved_doc_ids, index_time, retrieval_time
        finally:
            if index is not None:
                del index
            if temp_dir is not None:
                shutil.rmtree(temp_dir)
            gc.collect()

    def _compute_retrieval_metrics(self, query_ids_list, retrieved_doc_ids, ground_truth, top_k_values):
        """Compute retrieval metrics (Recall, NDCG, MRR) for retrieved documents.
        
        Args:
            query_ids_list: List of query IDs
            retrieved_doc_ids: List of retrieved document ID lists
            ground_truth: Ground truth relevance labels
            top_k_values: List of k values for metrics
            
        Returns:
            Tuple of (recall_sums, ndcg_sums, mrr_sum, valid_queries)
        """
        recall_sums = {k: 0.0 for k in top_k_values}
        ndcg_sums = {k: 0.0 for k in top_k_values}
        mrr_sum = 0.0
        valid_queries = 0
        
        if retrieved_doc_ids is None:
            return recall_sums, ndcg_sums, mrr_sum, valid_queries
        
        for query_id, retrieved_docs in zip(query_ids_list, retrieved_doc_ids):
            relevant_docs_list, relevant_docs_dict = get_relevant_docs_list(query_id, ground_truth)
            if not relevant_docs_list:
                continue

            valid_queries += 1
            for k in top_k_values:
                recall_sums[k] += calculate_recall_at_k(retrieved_docs, relevant_docs_list, k)
                ndcg_sums[k] += calculate_ndcg_at_k(retrieved_docs, relevant_docs_dict, k)
            mrr_sum += calculate_mrr(retrieved_docs, relevant_docs_list)
        
        return recall_sums, ndcg_sums, mrr_sum, valid_queries

    def _aggregate_distributed_metrics(self, recall_sums, ndcg_sums, mrr_sum, valid_queries, top_k_values):
        """Aggregate metrics from all processes in distributed mode.
        
        Returns:
            Tuple of (recall_sums, ndcg_sums, mrr_sum, valid_queries) aggregated across all processes
        """
        if not self.is_ddp:
            return recall_sums, ndcg_sums, mrr_sum, valid_queries
        
        gathered_stats = [None for _ in range(self.world_size)]
        dist.all_gather_object(
            gathered_stats,
            {
                "recall_sums": recall_sums,
                "ndcg_sums": ndcg_sums,
                "mrr_sum": mrr_sum,
                "valid_queries": valid_queries,
            },
            group=get_gloo_group(),
        )

        recall_sums_total = {k: 0.0 for k in top_k_values}
        ndcg_sums_total = {k: 0.0 for k in top_k_values}
        mrr_sum_total = 0.0
        valid_queries_total = 0
        for entry in gathered_stats:
            valid_queries_total += entry["valid_queries"]
            mrr_sum_total += entry["mrr_sum"]
            for k in top_k_values:
                recall_sums_total[k] += entry["recall_sums"][k]
                ndcg_sums_total[k] += entry["ndcg_sums"][k]

        return recall_sums_total, ndcg_sums_total, mrr_sum_total, valid_queries_total

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """Evaluate the model on the given dataset.
        
        This method performs the following steps:
        1. Encode all queries and passages
        2. Gather results from all processes (if distributed)
        3. Build index and retrieve top-k documents
        4. Compute retrieval metrics
        5. Aggregate metrics across processes (if distributed)
        """
        self._memory_tracker.start()
        start_time = time.time()
        self.model.eval()
        
        # Step 1: Encode all queries and passages
        loss_sum, num_batches, all_query_embeddings, all_passage_embeddings = self._encode_all(eval_dataset)
        encoding_time = time.time()
        encoding_time_usage = encoding_time - start_time
        
        # Step 2: Gather and merge results from all processes
        (
            loss, query_ids_list, query_embeddings, query_masks,
            passage_ids_list, passage_embeddings, passage_masks,
        ) = self._gather_encoding_results(loss_sum, num_batches, all_query_embeddings, all_passage_embeddings)
        del all_query_embeddings, all_passage_embeddings
        gc.collect()
        
        gathering_time = time.time()
        gathering_time_usage = gathering_time - encoding_time
        
        # Step 3: Build index and retrieve
        top_k_values = [1, 5, 10, 100]
        max_k = max(top_k_values)
        ground_truth = load_dataset_labels(eval_dataset or self.eval_dataset)
        
        retrieved_doc_ids, index_building_time_usage, retrieval_time_usage = self._build_index_and_retrieve(
            query_embeddings, query_masks,
            passage_embeddings, passage_masks,
            passage_ids_list, top_k=max_k
        )
        
        # Clean up embeddings after retrieval
        del query_embeddings, query_masks, passage_embeddings, passage_masks
        gc.collect()
        
        # Step 4: Compute retrieval metrics
        recall_sums, ndcg_sums, mrr_sum, valid_queries = self._compute_retrieval_metrics(
            query_ids_list, retrieved_doc_ids, ground_truth, top_k_values
        )
        del retrieved_doc_ids
        gc.collect()
        
        # Step 5: Aggregate metrics across processes
        recall_sums, ndcg_sums, mrr_sum, valid_queries = self._aggregate_distributed_metrics(
            recall_sums, ndcg_sums, mrr_sum, valid_queries, top_k_values
        )

        # Build final metrics dictionary
        retrieval_time = gathering_time + index_building_time_usage + retrieval_time_usage
        metrics = self._build_metrics_dict(
            metric_key_prefix, loss, valid_queries,
            recall_sums, ndcg_sums, mrr_sum, top_k_values,
            start_time, num_batches,
            encoding_time_usage, gathering_time_usage,
            index_building_time_usage, retrieval_time_usage
        )

        # Log and broadcast metrics
        if self.rank == 0:
            self.log(metrics)
            self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics=metrics)
            self._memory_tracker.stop_and_update_metrics(metrics)

        if self.is_ddp:
            broadcast_data = [metrics]
            dist.broadcast_object_list(broadcast_data, 0)
            return broadcast_data[0]
        return metrics

    def _build_metrics_dict(
        self, metric_key_prefix, loss, valid_queries,
        recall_sums, ndcg_sums, mrr_sum, top_k_values,
        start_time, num_batches,
        encoding_time_usage, gathering_time_usage,
        index_building_time_usage, retrieval_time_usage
    ):
        """Build the final metrics dictionary."""
        metrics = {}
        metrics[f"{metric_key_prefix}_loss"] = loss
        metrics[f"{metric_key_prefix}_num_queries"] = valid_queries
        
        if valid_queries > 0:
            metrics[f"{metric_key_prefix}_MRR@{max(top_k_values)}"] = mrr_sum / valid_queries
            for k in top_k_values:
                metrics[f"{metric_key_prefix}_Recall@{k}"] = recall_sums[k] / valid_queries
                metrics[f"{metric_key_prefix}_NDCG@{k}"] = ndcg_sums[k] / valid_queries

        # Add timing metrics
        metrics.update(speed_metrics(
            metric_key_prefix,
            start_time,
            num_samples=num_batches * self.args.per_device_eval_batch_size * self.world_size,
            num_steps=num_batches,
        ))
        metrics[f"{metric_key_prefix}_encoding_time"] = encoding_time_usage
        metrics[f"{metric_key_prefix}_gathering_time"] = gathering_time_usage
        metrics[f"{metric_key_prefix}_index_building_time"] = index_building_time_usage
        metrics[f"{metric_key_prefix}_retrieval_time"] = retrieval_time_usage
        metrics[f"{metric_key_prefix}_metrics_calculation_time"] = time.time() - start_time - encoding_time_usage - gathering_time_usage - index_building_time_usage - retrieval_time_usage
        
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_inputs = inputs["query_inputs"]
        passage_inputs = inputs["passage_inputs"]
        passage_ids = inputs["passage_ids_tensor"]
        outputs = model(
            query_inputs=query_inputs,
            passage_inputs=passage_inputs,
            passage_ids=passage_ids,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class DistilRetrievalTrainer(RetrievalTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        query_inputs = inputs["query_inputs"]
        passage_inputs = inputs["passage_inputs"]
        reranker_labels = inputs["reranker_labels"]
        passage_ids = inputs.get("passage_ids")
        scores = model(
            query_inputs=query_inputs,
            passage_inputs=passage_inputs,
            passage_ids=passage_ids,
        ).scores
        
        if model.is_ddp:
            reranker_labels = dist_gather_tensor(reranker_labels, model.world_size, model.process_rank)
        
        batch_size, total_passages = scores.size()
        num_labels = reranker_labels.size(1)
        start_idxs = torch.arange(0, batch_size * num_labels, num_labels, device=scores.device)
        idx_matrix = start_idxs.view(-1, 1) + torch.arange(num_labels, device=scores.device)
        student_scores = scores.gather(1, idx_matrix) # shape: (batch_size, num_labels)

        # Temperature‐scaled soft distributions
        T = self.args.distil_temperature
        student_log   = torch.log_softmax(student_scores.float() / T, dim=1)
        teacher_probs = torch.softmax(reranker_labels.float()    / T, dim=1)

        # KL Divergence loss
        loss = torch.nn.functional.kl_div(
            student_log,
            teacher_probs,
            reduction="batchmean"
        )

        return loss
