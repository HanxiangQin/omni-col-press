import logging
logger = logging.getLogger(__name__)
from typing import Dict, List
from src.arguments import DataArguments, EvaluateArguments
from src.dataset import EncodeDataset
from src.index import MultiModalIndex, FastPlaidIndex, MultiVecIndex
import jsonlines
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

def load_index(eval_args: EvaluateArguments, device: str) -> MultiModalIndex:
    index_path = eval_args.index_path
    if not index_path:
        raise ValueError("Please specify the index directory via --index_path")
    logger.info(f"Loading index from {index_path}...")
    index: MultiModalIndex = None
    if "fast-plaid" in eval_args.index_type:
        index = FastPlaidIndex.load(index_path, device)
    elif "multivec" in eval_args.index_type:
        index = MultiVecIndex.load(index_path, device)
    elif "flat" in eval_args.index_type:
        if eval_args.is_pickle:
            index = MultiModalIndex.load_from_pickle(index_path, device)
        else:
            index = MultiModalIndex.load(index_path, device)
    else:
        raise ValueError(f"Index type {eval_args.index_type} not supported")

    return index

def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    qrels = defaultdict(dict)

    # jsonl format
    try:
        with jsonlines.open(qrels_path) as f:
            for line in f:
                query_id = line['query_id']
                doc_id = line['doc_id']
                relevance = line['relevance']
                qrels[query_id][doc_id] = relevance
        
        return dict(qrels)
    except Exception as e:
        try:
            with open(qrels_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        query_id = parts[0]
                        doc_id = parts[2]
                        relevance = int(parts[3])
                        qrels[query_id][doc_id] = relevance
            return dict(qrels)
        except Exception as e:
            logger.error(f"Error loading qrels from {qrels_path}: {e}")
            raise e


def load_dataset_labels(dataset) -> Dict[str, List[str]]:
    labels = {}
    
    for item in dataset:
        if 'query' in item and 'passages' in item:
            query_id = item['query']['id']
            positive_docs = []
            for passage in item['passages']:
                positive_docs.append(passage['id'])
                # Only the first passage is the positive passage
                break
            
            if positive_docs:
                labels[query_id] = positive_docs
    
    return labels

def load_ground_truth(data_args: DataArguments, query_dataset: EncodeDataset) -> Dict[str, Dict[str, int]]:
    logger.info("Preparing ground truth...")
    if data_args.qrels_path:
        logger.info(f"Loading qrels from {data_args.qrels_path}...")
        ground_truth = load_qrels(data_args.qrels_path)
    elif data_args.use_dataset_labels:
        logger.info("Extracting labels from dataset...")
        ground_truth = {}
        labels = load_dataset_labels(query_dataset.encode_data)
        for query_id, relevant_docs in labels.items():
            ground_truth[query_id] = {doc: 1 for doc in relevant_docs}
    else:
        raise ValueError("Either --qrels_path or --use_dataset_labels must be specified for evaluation")
    logger.info(f"Loaded ground truth for {len(ground_truth)} queries")

    return ground_truth