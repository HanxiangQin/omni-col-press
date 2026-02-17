from typing import List, Dict, Any, Optional
import numpy as np
from src.index import MultiModalIndex, MultiVecIndex, FastPlaidIndex

def build_index(
    index_type: str, 
    embeddings: np.ndarray, 
    masks: np.ndarray, 
    ids: List[str], 
    metadata: Optional[List[Dict[str, Any]]] = None, 
    device: str = "cuda",
    build_config: Optional[Dict] = None, 
    index_output_path: Optional[str] = None
) -> MultiModalIndex:
    # embeddings can be [N, D] (single-vector) or [N, S, D] (multi-vector)
    embedding_dim = embeddings.shape[-1]
    index: MultiModalIndex
    if "multivec" in index_type:
        index = MultiVecIndex(
            device=device,
            index_type=index_type,
            build_config=build_config
        )
    elif "fast-plaid" in index_type:
        index = FastPlaidIndex(
            index_path=index_output_path,
            index_type=index_type,
            override=True,
            build_config=build_config,
            device=device
        )
    elif "flat" in index_type:
        index = MultiModalIndex(
            dim=embedding_dim,
            index_type=index_type,
            build_config=build_config
        )
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    index.add_vectors(doc_ids=ids, vectors=embeddings, masks=masks, metadata=metadata)
    return index