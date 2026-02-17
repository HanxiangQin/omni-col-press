from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from src.index.base_index import MultiModalIndex
from src.utils import colbert_sim
import torch
import os
import pickle
import logging

logger = logging.getLogger(__name__)

class MultiVecIndex(MultiModalIndex):
    def __init__(self,
            device: str = "cuda",
            index_type: str = "multivec",
            build_config: Optional[Dict] = None):
        self.device = device
        self.build_config = build_config or {}
        self.index_type = index_type
        self.tensor_index = None
        self.mask_index = None  # Store masks for accurate position tracking
        self.doc_ids = []
        self.metadata = {}

    def _compute_similarity(
        self,
        query: torch.Tensor,    # [B,Q,D]
        doc: torch.Tensor,      # [B,S,D]
        query_mask: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
        normalize: bool = False,
        return_argmax: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return colbert_sim(query, doc, query_mask, doc_mask, normalize, return_argmax, use_full_precision=True)
    
    def add_vectors(
        self,
        doc_ids: List[str],
        vectors: Union[np.ndarray, torch.Tensor, List],
        masks: Union[np.ndarray, torch.Tensor, List] = None,
        metadata: Optional[List[Dict]] = None
    ) -> "MultiVecIndex":
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)
        vectors = vectors.to(device=self.device)

        if self.tensor_index is None:
            self.tensor_index = vectors
        else:
            self.tensor_index = torch.cat([self.tensor_index, vectors], dim=0)
        
        # Store masks for accurate position tracking
        if masks is not None:
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks)
            masks = masks.to(device=self.device)

            if self.mask_index is None:
                self.mask_index = masks
            else:
                self.mask_index = torch.cat([self.mask_index, masks], dim=0)
        
        start_idx = len(self.doc_ids)
        self.doc_ids.extend(doc_ids)
        
        if metadata:
            for i, meta in enumerate(metadata):
                self.metadata[start_idx + i] = meta

        return self

    def _search_torch(self, query_vectors: torch.Tensor, query_mask: Optional[torch.Tensor] = None, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        query_vectors = query_vectors.unsqueeze(0)
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(0)
        return self._batch_search_torch(query_vectors, query_mask, k)

    def _batch_search_torch(self, query_vectors: torch.Tensor, query_mask: Optional[torch.Tensor] = None, k: int = 10, batch_size: int = 1000, quiet: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure queries reside on the same device/dtype as the index to avoid implicit transfers
        query_vectors = query_vectors.to(device=self.tensor_index.device, dtype=self.tensor_index.dtype)
        if query_mask is not None:
            query_mask = query_mask.to(device=self.tensor_index.device)

        topk_scores_chunks: List[torch.Tensor] = []
        topk_indices_chunks: List[torch.Tensor] = []

        num_docs = self.tensor_index.shape[0]
        k = min(k, num_docs)

        num_queries = query_vectors.shape[0]
        for start in range(0, num_queries, batch_size):
            end = min(start + batch_size, num_queries)
            batch_queries = query_vectors[start:end]
            batch_mask = query_mask[start:end] if query_mask is not None else None

            scores = self._compute_similarity(batch_queries, self.tensor_index, query_mask=batch_mask, doc_mask=self.mask_index)
            batch_scores, batch_indices = torch.topk(scores, k, dim=1, largest=True, sorted=True)

            topk_scores_chunks.append(batch_scores)
            topk_indices_chunks.append(batch_indices)

        topk_scores = torch.cat(topk_scores_chunks, dim=0)
        topk_indices = torch.cat(topk_indices_chunks, dim=0)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()
    
    def _search_numpy(self, query_vectors: np.ndarray, query_mask: Optional[np.ndarray] = None, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        query_vectors = np.expand_dims(query_vectors, axis=0)
        if query_mask is not None:
            query_mask = np.expand_dims(query_mask, axis=0)
        return self.batch_search(query_vectors, query_mask, k)

    def _batch_search_numpy(self, query_vectors: np.ndarray, query_mask: Optional[np.ndarray] = None, k: int = 10, batch_size: int = 1000, quiet: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        tensor_query_vectors = torch.from_numpy(query_vectors).to(dtype=self.tensor_index.dtype, device=self.tensor_index.device)
        tensor_query_mask = None
        if query_mask is not None:
            tensor_query_mask = torch.from_numpy(query_mask).to(device=self.tensor_index.device)
        return self._batch_search_torch(tensor_query_vectors, tensor_query_mask, k, batch_size, quiet)

    def search(self, query_vectors: np.ndarray | torch.Tensor, query_mask: Optional[np.ndarray | torch.Tensor] = None, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(query_vectors, np.ndarray):
            return self._search_numpy(query_vectors, query_mask, k)
        elif isinstance(query_vectors, torch.Tensor):
            return self._search_torch(query_vectors, query_mask, k)
        else:
            raise ValueError(f"Unsupported query vector type: {type(query_vectors)}")
    
    def batch_search(self, query_vectors: np.ndarray | torch.Tensor, query_mask: Optional[np.ndarray | torch.Tensor] = None, k: int = 10, batch_size: int = 1000, quiet: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(query_vectors, np.ndarray):
            return self._batch_search_numpy(query_vectors, query_mask, k, batch_size, quiet)
        elif isinstance(query_vectors, torch.Tensor):
            return self._batch_search_torch(query_vectors, query_mask, k, batch_size, quiet)
        else:
            raise ValueError(f"Unsupported query vector type: {type(query_vectors)}")

    def save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        
        index_file = os.path.join(save_path, "index.pt")
        torch.save(self.tensor_index, index_file)
        
        # Save masks if available
        if self.mask_index is not None:
            mask_file = os.path.join(save_path, "masks.pt")
            torch.save(self.mask_index, mask_file)
            logger.info(f"Saved masks with shape {self.mask_index.shape} to {mask_file}")
        
        metadata_file = os.path.join(save_path, "metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'doc_ids': self.doc_ids,
                'metadata': self.metadata,
                'build_config': self.build_config,
                'index_type': self.index_type,
                'has_masks': self.mask_index is not None,
            }, f)
        
    @classmethod
    def load(cls, metadata_path: str, device: str = "cuda"):
        try:
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
            logger.info(f"Loading index from metadata: {metadata_path}, local path: {local_path}")
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            for file in os.listdir(local_path):
                if file.endswith("index.pt"):
                    index_file = os.path.join(local_path, file)
                    break
            tensor_index = torch.load(index_file, map_location=device)
            
            index_type = metadata['index_type'] if 'index_type' in metadata else "multivec"
            instance = cls(index_type=index_type, build_config=metadata.get('build_config', {}))
            instance.tensor_index = tensor_index
            # Load masks if available
            if metadata.get('has_masks', False):
                mask_file = os.path.join(local_path, "masks.pt")
                if os.path.exists(mask_file):
                    instance.mask_index = torch.load(mask_file, map_location=device)
                    logger.info(f"Loaded masks with shape {instance.mask_index.shape}")
                else:
                    logger.warning(f"Masks file not found at {mask_file}, positions may be inaccurate")
            instance.doc_ids = metadata.get('doc_ids', [])
            instance.metadata = metadata.get('metadata', {})
            instance.device = device
            
            return instance
        except Exception as e:
            logger.error(f"Error loading index from {metadata_path}: {e}")
            raise e

    def get_stats(self) -> Dict:
        index_size = 1
        for dim in self.tensor_index.shape:
            index_size *= dim
        index_size *= 4
        index_size /= 1024 * 1024
        
        mask_size_mb = 0
        if self.mask_index is not None:
            mask_size = 1
            for dim in self.mask_index.shape:
                mask_size *= dim
            mask_size_mb = mask_size * 4 / 1024 / 1024  # Assuming float32
        
        return {
            'index_class': self.__class__.__name__,
            'index_type': self.index_type,
            'num_docs': len(self.doc_ids),
            'dim': self.tensor_index.shape,
            'index_size_mb': index_size,
            'has_masks': self.mask_index is not None,
            'mask_size_mb': mask_size_mb,
            'build_config': self.build_config
        }

    def get_sample_vector(self) -> List[torch.Tensor]:
        sample_vectors = []
        for i in range(10):
            sample_vectors.append(self.tensor_index[i])
        return sample_vectors

    def get_dim(self) -> str:
        return f"{self.tensor_index.shape}"