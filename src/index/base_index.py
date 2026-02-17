import os
import pickle
import json
import faiss
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from tqdm import tqdm
import logging
import torch

logger = logging.getLogger(__name__)


class MultiModalIndex:    
    def __init__(self, 
                 dim: int, 
                 index_type: str = "flat",
                 build_config: Optional[Dict] = None,
                 factory_str: Optional[str] = None):
        self.dim = dim
        self.index_type = index_type
        self.build_config = build_config or {}
        
        self.index = self._create_index(dim, index_type)
        self.index.verbose = True
        self.is_trained = False
        self.doc_ids = []
        self.metadata = {}

    def _create_index(self, dim: int, index_type: str):
        if index_type == "flat":
            index = faiss.IndexFlatIP(dim)
        # elif index_type == "ivf":
        #     # IVF index with 1024 clusters
        #     quantizer = faiss.IndexFlatIP(dim)
        #     index = faiss.IndexIVFFlat(quantizer, dim, 1024, faiss.METRIC_INNER_PRODUCT)
        # elif index_type == "hnsw":
        #     # HNSW index
        #     index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        # elif index_type == "custom" and factory_str:
        #     index = faiss.index_factory(dim, factory_str)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        return index
        
    def _train(self, training_vectors: np.ndarray):
        if not self.index.is_trained:
            logger.info(f"Training index with {len(training_vectors)} vectors...")
            self.index.train(training_vectors.astype(np.float32))
            self.is_trained = True
        else:
            self.is_trained = True
            
    def add_vectors(self, doc_ids: List[str], vectors: np.ndarray, masks: np.ndarray = None, metadata: Optional[List[Dict]] = None) -> "MultiModalIndex":
        vectors = vectors.astype(np.float32)
        if not self.is_trained:
            self._train(vectors)
            
        start_idx = len(self.doc_ids)
        self.index.add(vectors)
        self.doc_ids.extend(doc_ids)
        
        if metadata:
            for i, meta in enumerate(metadata):
                self.metadata[start_idx + i] = meta

        logger.info(f"Added {len(vectors)} vectors to index. Total: {self.index.ntotal}")
        return self
        
    def _search_numpy(self, query_vectors: np.ndarray, query_mask: Optional[np.ndarray] = None, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        return self.index.search(query_vectors, k)
        
    def _batch_search_numpy(self, 
                    query_vectors: np.ndarray, 
                    query_mask: Optional[np.ndarray] = None,
                    k: int = 10, 
                    batch_size: int = 1000,
                    quiet: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        num_queries = query_vectors.shape[0]
        topk_scores = []
        topk_indices = []
        
        for start_idx in tqdm(range(0, num_queries, batch_size), disable=quiet, desc="Searching"):
            end_idx = min(start_idx + batch_size, num_queries)
            batch_queries = query_vectors[start_idx:end_idx]
            batch_query_mask = query_mask[start_idx:end_idx] if query_mask is not None else None
            
            scores, indices = self._search_numpy(batch_queries, batch_query_mask, k)
            topk_scores.append(scores)
            topk_indices.append(indices)
            
        topk_scores = np.concatenate(topk_scores, axis=0)
        topk_indices = np.concatenate(topk_indices, axis=0)
        
        return topk_scores, topk_indices

    def _search_torch(self, query_vectors: torch.Tensor, query_mask: Optional[torch.Tensor] = None, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        query_vectors = query_vectors.unsqueeze(0)
        return self._batch_search_torch(query_vectors, query_mask, k)

    def _batch_search_torch(self, query_vectors: torch.Tensor, query_mask: Optional[torch.Tensor] = None, k: int = 10, batch_size: int = 1000, quiet: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        numpy_query_vectors = query_vectors.cpu().numpy()
        return self._batch_search_numpy(numpy_query_vectors, query_mask, k, batch_size, quiet)

    def search(self, query_vectors: np.ndarray | torch.Tensor, query_mask: Optional[np.ndarray | torch.Tensor] = None, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        # Note: query_mask is ignored for FAISS-based single-vector indices
        if isinstance(query_vectors, np.ndarray):
            return self._search_numpy(query_vectors, query_mask, k)
        elif isinstance(query_vectors, torch.Tensor):
            return self._search_torch(query_vectors, query_mask, k)
        else:
            raise ValueError(f"Unsupported query vector type: {type(query_vectors)}")

    def batch_search(self, query_vectors: np.ndarray | torch.Tensor, query_mask: Optional[np.ndarray | torch.Tensor] = None, k: int = 10, batch_size: int = 1000, quiet: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # Note: query_mask is ignored for FAISS-based single-vector indices
        if isinstance(query_vectors, np.ndarray):
            return self._batch_search_numpy(query_vectors, query_mask, k, batch_size, quiet)
        elif isinstance(query_vectors, torch.Tensor):
            return self._batch_search_torch(query_vectors, query_mask, k, batch_size, quiet)
        else:
            raise ValueError(f"Unsupported query vector type: {type(query_vectors)}")
        
    def get_doc_ids(self, indices: np.ndarray) -> List[List[str]]:
        results = []
        for query_indices in indices:
            doc_ids = [self.doc_ids[idx] if idx >= 0 else "" for idx in query_indices]
            results.append(doc_ids)
        return results
        
    def get_metadata(self, indices: np.ndarray) -> List[List[Dict]]:
        results = []
        for query_indices in indices:
            metadata = [self.metadata.get(idx, {}) if idx >= 0 else {} for idx in query_indices]
            results.append(metadata)
        return results
        
    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        index_file = os.path.join(save_path, "index.faiss")
        faiss.write_index(self.index, index_file)
        
        # Save metadata
        metadata_file = os.path.join(save_path, "metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'doc_ids': self.doc_ids,
                'metadata': self.metadata,
                'dim': self.dim,
                'index_type': self.index_type,
                'build_config': self.build_config,
                'is_trained': self.is_trained
            }, f)
            
        logger.info(f"Index saved to {save_path}")

    def save_pickle(self, save_path: str):
        all_retrieved_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        with open(save_path, 'wb') as f:
            pickle.dump((all_retrieved_vectors, self.doc_ids), f)
        
    @classmethod
    def load(cls, metadata_path: str, device: str = "cpu"):
        try:
            # Load config
            if os.path.isfile(metadata_path):
                local_path = os.path.dirname(metadata_path)
                
            elif os.path.isdir(metadata_path):
                local_path = metadata_path
                for file in os.listdir(local_path):
                    if file.endswith("metadata.pkl"):
                        metadata_path = os.path.join(local_path, file)
                        break
            else:
                raise ValueError(f"Invalid metadata path: {metadata_path}")
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            # Create instance
            instance = cls(
                dim=metadata.get('dim', 2048),
                index_type=metadata.get('index_type', 'flat'),
                build_config=metadata.get('build_config', {})
            )
            
            # Load FAISS index
            for file in os.listdir(local_path):
                if file.endswith("index.faiss"):
                    index_file = os.path.join(local_path, file)
                    break
            instance.index = faiss.read_index(index_file)
            
            instance.doc_ids = metadata.get('doc_ids', [])
            instance.metadata = metadata.get('metadata', {})
            instance.is_trained = metadata.get('is_trained', False)

            logger.info(f"Index loaded from {local_path} with {len(instance.doc_ids)} documents")
            return instance
        except Exception as e:
            logger.error(f"Error loading index from {metadata_path}: {e}")
            raise e

    @classmethod
    def load_from_pickle(cls, pickle_path: str, device: str = "cpu"):
        with open(pickle_path, 'rb') as f:
            encoded_embeddings, lookup_indices = pickle.load(f)
        instance = cls(
            dim=encoded_embeddings.shape[-1],
            index_type="flat",
            build_config=None
        )
        instance.add_vectors(encoded_embeddings, lookup_indices)
        return instance
        
    def get_stats(self) -> Dict:
        return {
            'index_class': self.__class__.__name__,
            'index_type': self.index_type,
            'num_docs': len(self.doc_ids),
            'dim': self.dim,
            'is_trained': self.is_trained,
            'index_size_mb': self.index.ntotal * self.dim * 4 / (1024 * 1024),
            'build_config': self.build_config
        }

    def get_dim(self) -> str:
        return f"{self.index.d}"