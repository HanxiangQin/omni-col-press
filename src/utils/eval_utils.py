from typing import Dict, List
import numpy as np

def calculate_recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    if not relevant_docs:
        return 0.0
    
    retrieved_k = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    return len(retrieved_k & relevant_set) / len(relevant_set)


def calculate_mrr(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    if not relevant_docs:
        return 0.0
    
    relevant_set = set(relevant_docs)
    
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_set:
            return 1.0 / (i + 1)
    
    return 0.0


def calculate_ndcg_at_k(retrieved_docs: List[str], relevant_docs: Dict[str, int], k: int) -> float:
    if not relevant_docs:
        return 0.0
    
    # DCG@k
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k], start=1):
        relevance = relevant_docs.get(doc, 0)
        if relevance > 0:
            dcg += relevance / np.log2(i + 1)
    
    # IDCG@k
    sorted_relevances = sorted(relevant_docs.values(), reverse=True)
    idcg = 0.0
    for i, relevance in enumerate(sorted_relevances[:k], start=1):
        if relevance > 0:
            idcg += relevance / np.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return float(dcg / idcg)

def get_relevant_docs_list(query_id: str, labels: Dict[str, List[str]]):
    if query_id not in labels:
        return None, None
    dict_key = labels[query_id]
    if isinstance(dict_key, list):
        relevant_docs_list = dict_key
        relevant_docs_dict = {doc: 1 for doc in relevant_docs_list}
    elif isinstance(dict_key, dict):
        relevant_docs_list = [doc for doc, rel in dict_key.items() if rel > 0]
        relevant_docs_dict = dict_key
    return relevant_docs_list, relevant_docs_dict