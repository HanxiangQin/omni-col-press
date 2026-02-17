from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Union
import torch
import numpy as np
import os
import logging
import pickle

from sqlitedict import SqliteDict
from fast_plaid import search
from src.index.base_index import MultiModalIndex

logger = logging.getLogger(__name__)


def convert_embeddings_to_torch(
    embeddings: np.ndarray | torch.Tensor | list,
) -> list[torch.Tensor]:
    """Convert embeddings to list of torch tensors as expected by fast-plaid."""
    if isinstance(embeddings, list):
        if len(embeddings) == 0:
            return []
        if isinstance(embeddings[0], torch.Tensor):
            return [embedding.float() for embedding in embeddings]
        elif isinstance(embeddings[0], np.ndarray):
            return [torch.from_numpy(emb).float() for emb in embeddings]

    if isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 3:  # batch_size, n_tokens, embedding_size
            return [torch.from_numpy(embeddings[i]).float() for i in range(embeddings.shape[0])]
        elif len(embeddings.shape) == 2:  # n_tokens, embedding_size
            return [torch.from_numpy(embeddings).float()]

    if isinstance(embeddings, torch.Tensor):
        if len(embeddings.shape) == 3:  # batch_size, n_tokens, embedding_size
            return [embeddings[i].float() for i in range(embeddings.shape[0])]
        elif len(embeddings.shape) == 2:  # n_tokens, embedding_size
            return [embeddings.float()]

    return embeddings


def convert_masks_to_torch(
    masks: np.ndarray | torch.Tensor | list | None,
) -> list[torch.Tensor] | None:
    """Convert masks to list of torch bool tensors for trimming."""
    if masks is None:
        return None
    
    if isinstance(masks, list):
        if len(masks) == 0:
            return []
        if isinstance(masks[0], torch.Tensor):
            return [m.bool() for m in masks]
        elif isinstance(masks[0], np.ndarray):
            return [torch.from_numpy(m).bool() for m in masks]

    if isinstance(masks, np.ndarray):
        if len(masks.shape) == 2:  # batch_size, n_tokens
            return [torch.from_numpy(masks[i]).bool() for i in range(masks.shape[0])]
        elif len(masks.shape) == 1:  # n_tokens (single mask)
            return [torch.from_numpy(masks).bool()]

    if isinstance(masks, torch.Tensor):
        if len(masks.shape) == 2:  # batch_size, n_tokens
            return [masks[i].bool() for i in range(masks.shape[0])]
        elif len(masks.shape) == 1:  # n_tokens (single mask)
            return [masks.bool()]

    return None


class FastPlaidIndex(MultiModalIndex):
    """Multi-vector index using fast-plaid backend for high-performance late interaction search.
    
    This index supports multi-vector representations where each document/query is represented
    by multiple token embeddings, enabling fine-grained token-level similarity matching.
    
    Parameters
    ----------
    index_path : str
        The folder where the index will be stored.
    index_type : str
        The name of the index.
    override : bool
        Whether to override the collection if it already exists.
    nbits : int
        The number of bits to use for product quantization. Valid values: 1, 2, 4, or 8.
    kmeans_niters : int
        The number of iterations for K-means during index creation.
    max_points_per_centroid : int
        Maximum points per centroid during K-means.
    n_ivf_probe : int
        Number of IVF probes during search (higher = better recall, slower).
    n_full_scores : int
        Number of candidates for full scoring (higher = better accuracy, slower).
    n_samples_kmeans : int | None
        Number of samples for K-means clustering.
    batch_size : int
        Internal batch size for processing queries.
    show_progress : bool
        Whether to display progress bars during operations.
    device : str | list[str] | None
        Device(s) to use for computation.
    use_triton : bool | None
        Whether to use Triton kernels for K-means.
    """

    def __init__(
        self,
        index_path: str = "indexes",
        index_type: str = "fast-plaid",
        build_config: Optional[Dict] = None,
        override: bool = False,
        nbits: int = 8,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        n_ivf_probe: int = 64,
        n_full_scores: int = 8192,
        n_samples_kmeans: int | None = None,
        batch_size: int = 1 << 18,
        show_progress: bool = True,
        device: str | list[str] | None = None,
        use_triton: bool | None = None,
    ) -> None:
        self.index_path = index_path
        self.index_type = index_type
        self.build_config = build_config or {}
        self.nbits = nbits
        self.kmeans_niters = kmeans_niters
        self.max_points_per_centroid = max_points_per_centroid
        self.n_ivf_probe = n_ivf_probe
        self.n_full_scores = n_full_scores
        self.n_samples_kmeans = n_samples_kmeans
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.device = device
        self.use_triton = use_triton

        self.fast_plaid_index_path = os.path.join(self.index_path, "fast_plaid_index")
        os.makedirs(self.index_path, exist_ok=True)
        os.makedirs(self.fast_plaid_index_path, exist_ok=True)

        # SQLite mappings for document IDs
        self.doc_ids_to_plaid_ids_path = os.path.join(
            self.index_path, "doc_ids_to_plaid_ids.sqlite"
        )
        self.plaid_ids_to_doc_ids_path = os.path.join(
            self.index_path, "plaid_ids_to_doc_ids.sqlite"
        )

        # Initialize or load the fast-plaid index
        self.fast_plaid = search.FastPlaid(
            index=self.fast_plaid_index_path, device=device
        )

        if override:
            # Remove existing SQLite mappings
            if os.path.exists(self.doc_ids_to_plaid_ids_path):
                os.remove(self.doc_ids_to_plaid_ids_path)
            if os.path.exists(self.plaid_ids_to_doc_ids_path):
                os.remove(self.plaid_ids_to_doc_ids_path)
            self.is_indexed = False
        else:
            # Check if index already exists
            doc_ids_to_plaid_ids = self._load_doc_ids_to_plaid_ids()
            self.is_indexed = len(doc_ids_to_plaid_ids) > 0
            doc_ids_to_plaid_ids.close()

    def _load_doc_ids_to_plaid_ids(self) -> SqliteDict:
        """Load the SQLite database that maps document IDs to PLAID IDs."""
        return SqliteDict(self.doc_ids_to_plaid_ids_path, outer_stack=False)

    def _load_plaid_ids_to_doc_ids(self) -> SqliteDict:
        """Load the SQLite database that maps PLAID IDs to document IDs."""
        return SqliteDict(self.plaid_ids_to_doc_ids_path, outer_stack=False)

    def _trim_vectors(self, vectors: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        if masks is None:
            return vectors
        trimmed = []
        for vector, mask in zip(vectors, masks):
            trimmed.append(vector[mask.bool()])
        return trimmed

    def add_vectors(
        self,
        doc_ids: List[str],
        vectors: Union[np.ndarray, torch.Tensor, List],
        masks: Union[np.ndarray, torch.Tensor, List] = None,
        metadata: Optional[List[Dict]] = None
    ) -> "FastPlaidIndex":
        """Add document vectors to the index.
        
        Parameters
        ----------
        vectors : Union[np.ndarray, torch.Tensor, List]
            Document embeddings. Can be:
            - List of tensors/arrays, each with shape (n_tokens, embedding_dim)
            - 3D array/tensor with shape (n_docs, n_tokens, embedding_dim)
        masks : Union[np.ndarray, torch.Tensor, List]
            Document masks. Can be:
            - Only support numpy array for now for simplicity
            May support other formats in the future:
            - List of tensors/arrays, each with shape (n_tokens,)
            - 2D array/tensor with shape (n_docs, n_tokens)
        doc_ids : List[str]
            List of document IDs corresponding to each embedding.
        metadata : Optional[List[Dict]]
            Optional metadata for each document (stored in fast-plaid's metadata system).
            
        Returns
        -------
        FastPlaidIndex
            Self for method chaining.
        """
        # Convert embeddings to torch tensors
        vectors_torch = convert_embeddings_to_torch(vectors)
        # Trim document vectors using mask to remove padding tokens
        if masks is not None:
            masks_torch_list = convert_masks_to_torch(masks)
            vectors_torch = self._trim_vectors(vectors_torch, masks_torch_list)
        # Load SQLite mappings
        doc_ids_to_plaid_ids = self._load_doc_ids_to_plaid_ids()
        plaid_ids_to_doc_ids = self._load_plaid_ids_to_doc_ids()

        # Get the current number of documents for ID assignment
        current_max_id = (
            max([int(k) for k in plaid_ids_to_doc_ids.keys()])
            if plaid_ids_to_doc_ids
            else -1
        )

        if not self.is_indexed:
            # Create new index
            self.fast_plaid.create(
                documents_embeddings=vectors_torch,
                kmeans_niters=self.kmeans_niters,
                max_points_per_centroid=self.max_points_per_centroid,
                nbits=self.nbits,
                n_samples_kmeans=self.n_samples_kmeans,
                use_triton_kmeans=self.use_triton,
                metadata=metadata,
            )
            plaid_ids = list(range(len(vectors_torch)))
            self.is_indexed = True
        else:
            # Update existing index
            logger.warning(
                "Adding documents to existing index. This uses fast-plaid's update method "
                "which does not recompute centroids and may result in slightly lower accuracy."
            )
            self.fast_plaid.update(
                documents_embeddings=vectors_torch,
                metadata=metadata,
            )
            # Assign new plaid IDs starting from current_max_id + 1
            plaid_ids = list(
                range(
                    current_max_id + 1,
                    current_max_id + 1 + len(vectors_torch),
                )
            )

        # Store mappings
        for plaid_id, doc_id in zip(plaid_ids, doc_ids):
            doc_ids_to_plaid_ids[doc_id] = plaid_id
            plaid_ids_to_doc_ids[plaid_id] = doc_id

        doc_ids_to_plaid_ids.commit()
        doc_ids_to_plaid_ids.close()

        plaid_ids_to_doc_ids.commit()
        plaid_ids_to_doc_ids.close()

        logger.info(f"Added {len(vectors_torch)} documents to index. Total: {len(plaid_ids)}")
        return self

    def _convert_search_results(
        self,
        search_results: List[List[Tuple[int, float]]],
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert fast-plaid search results to (scores, indices) arrays.
        
        Parameters
        ----------
        search_results : List[List[Tuple[int, float]]]
            Results from fast-plaid search, format: list of lists of (plaid_id, score) tuples.
        k : int
            Number of nearest neighbors to return per query.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (scores, indices) arrays with shape (n_queries, k).
        """
        n_queries = len(search_results)
        scores = np.full((n_queries, k), -np.inf, dtype=np.float32)
        indices = np.full((n_queries, k), -1, dtype=np.int64)
        
        for q_idx, query_results in enumerate(search_results):
            # Limit results to k in case search returned more
            for r_idx, (plaid_id, score) in enumerate(query_results[:k]):
                scores[q_idx, r_idx] = score
                indices[q_idx, r_idx] = plaid_id
        
        return scores, indices

    def _search_impl(
        self,
        query_vectors: List[torch.Tensor],
        batch_size: int = 1000,
        k: int = 10,
        show_progress: bool | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Internal search implementation.
        
        Parameters
        ----------
        query_vectors : List[torch.Tensor]
            List of query embeddings as torch tensors.
        k : int
            Number of nearest neighbors to return.
        show_progress : bool | None
            Whether to show progress bar (uses instance default if None).
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (scores, indices) arrays with shape (n_queries, k).
        """
        if not self.is_indexed:
            raise ValueError("The index is empty. Please add documents before searching.")
        
        if show_progress is None:
            show_progress = self.show_progress

        n_ivf_probe = max(self.n_ivf_probe, k)
            
        # Perform search using fast-plaid
        search_results = self.fast_plaid.search(
            queries_embeddings=query_vectors,
            top_k=k,
            batch_size=self.batch_size,
            n_ivf_probe=n_ivf_probe,
            n_full_scores=self.n_full_scores,
            show_progress=show_progress,
        )
        return self._convert_search_results(search_results, k)

    def search(
        self,
        query_vectors: np.ndarray | torch.Tensor,
        query_mask: Optional[np.ndarray | torch.Tensor] = None,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        query_vectors_list = convert_embeddings_to_torch(query_vectors)
        # Trim query vectors using mask to remove padding tokens
        if query_mask is not None:
            query_mask_list = convert_masks_to_torch(query_mask)
            query_vectors_list = self._trim_vectors(query_vectors_list, query_mask_list)
        return self._search_impl(query_vectors_list, k=k)
    
    def batch_search(
        self,
        query_vectors: np.ndarray | torch.Tensor,
        query_mask: Optional[np.ndarray | torch.Tensor] = None,
        k: int = 10,
        batch_size: int = 1000,
        quiet: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        query_vectors_list = convert_embeddings_to_torch(query_vectors)
        # Trim query vectors using mask to remove padding tokens
        if query_mask is not None:
            query_mask_list = convert_masks_to_torch(query_mask)
            query_vectors_list = self._trim_vectors(query_vectors_list, query_mask_list)
        return self._search_impl(query_vectors_list, batch_size, k, not quiet)

    def get_doc_ids(self, indices: np.ndarray) -> List[List[str]]:
        plaid_ids_to_doc_ids = self._load_plaid_ids_to_doc_ids()
        
        results = []
        for query_indices in indices:
            doc_ids = []
            for idx in query_indices:
                idx_int = int(idx)
                if idx_int >= 0 and idx_int in plaid_ids_to_doc_ids:
                    doc_ids.append(plaid_ids_to_doc_ids[idx_int])
                else:
                    doc_ids.append("")
            results.append(doc_ids)
        
        plaid_ids_to_doc_ids.close()
        return results

    def save(self, save_path: str):
        """Save the index to disk.
        
        Note: fast-plaid automatically persists the index to the configured path.
        This method is provided for API compatibility and can be used to copy
        the index to an additional location.
        
        Parameters
        ----------
        save_path : str
            Directory path to save the index.
        """
        metadata_file = os.path.join(save_path, "metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'nbits': self.nbits,
                'kmeans_niters': self.kmeans_niters,
                'max_points_per_centroid': self.max_points_per_centroid,
                'n_ivf_probe': self.n_ivf_probe,
                'n_full_scores': self.n_full_scores,
                'n_samples_kmeans': self.n_samples_kmeans,
                'batch_size': self.batch_size,
                'show_progress': self.show_progress,
                'device': self.device,
                'use_triton': self.use_triton,
                'index_type': self.index_type,
                'build_config': self.build_config,
                'is_indexed': self.is_indexed
            }, f)
        if save_path != self.index_path:
            import shutil
            os.makedirs(save_path, exist_ok=True)
            # Copy the fast-plaid index and SQLite files
            if os.path.exists(self.fast_plaid_index_path):
                dest_fast_plaid = os.path.join(save_path, "fast_plaid_index")
                if os.path.exists(dest_fast_plaid):
                    shutil.rmtree(dest_fast_plaid)
                shutil.copytree(self.fast_plaid_index_path, dest_fast_plaid)
            if os.path.exists(self.doc_ids_to_plaid_ids_path):
                shutil.copy2(self.doc_ids_to_plaid_ids_path, save_path)
            if os.path.exists(self.plaid_ids_to_doc_ids_path):
                shutil.copy2(self.plaid_ids_to_doc_ids_path, save_path)
        logger.info(f"Index saved to {save_path}")
        
    @classmethod
    def load(cls, index_path: str, device: str | list[str] | None = None) -> "FastPlaidIndex":
        """Load an index from disk.
        
        Parameters
        ----------
        index_path : str
            Path to the index directory.
        device : str | list[str] | None
            Device(s) to use for computation.
            
        Returns
        -------
        FastPlaidIndex
            Loaded index instance.
        """
        metadata_file = os.path.join(index_path, "metadata.pkl")
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
            index_type = metadata.get('index_type', 'fast-plaid')
            nbits = metadata.get('nbits', 8)
            kmeans_niters = metadata.get('kmeans_niters', 4)
            max_points_per_centroid = metadata.get('max_points_per_centroid', 256)
            n_ivf_probe = metadata.get('n_ivf_probe', 64)
            n_full_scores = metadata.get('n_full_scores', 8192)
            n_samples_kmeans = metadata.get('n_samples_kmeans', None)
            batch_size = metadata.get('batch_size', 1 << 18)
            show_progress = metadata.get('show_progress', True)
            device = metadata.get('device', None)
            use_triton = metadata.get('use_triton', None)
            is_indexed = metadata.get('is_indexed', False)
            build_config = metadata.get('build_config', {})

        instance = cls(
            index_path=index_path,
            index_type=index_type,
            build_config=build_config,
            override=False,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            max_points_per_centroid=max_points_per_centroid,
            n_ivf_probe=n_ivf_probe,
            n_full_scores=n_full_scores,
            n_samples_kmeans=n_samples_kmeans,
            batch_size=batch_size,
            show_progress=show_progress,
            device=device,
            use_triton=use_triton
        )
        
        logger.info(f"Index loaded from {index_path}")
        return instance

    def get_stats(self) -> Dict:
        plaid_ids_to_doc_ids = self._load_plaid_ids_to_doc_ids()
        num_docs = len(plaid_ids_to_doc_ids)
        plaid_ids_to_doc_ids.close()
        
        return {
            'index_class': self.__class__.__name__,
            'index_type': self.index_type,
            'num_docs': num_docs,
            'nbits': self.nbits,
            'n_ivf_probe': self.n_ivf_probe,
            'n_full_scores': self.n_full_scores,
            'is_indexed': self.is_indexed,
            'index_path': self.index_path,
            'build_config': self.build_config
        }

    def get_sample_vector(self) -> List[torch.Tensor]:
        raise NotImplementedError(
            "Fast-plaid does not provide direct access to document embeddings. "
            "The embeddings are stored in compressed/quantized form and cannot be retrieved."
        )

    def remove_documents(self, doc_ids: List[str]) -> "FastPlaidIndex":
        """Remove documents from the index.
        
        Parameters
        ----------
        doc_ids : List[str]
            List of document IDs to remove.
            
        Returns
        -------
        FastPlaidIndex
            Self for method chaining.
            
        Note
        ----
        After deletion, remaining documents are re-indexed by fast-plaid to maintain
        sequential order. All documents with id > k will have their id decreased by 1.
        """
        doc_ids_to_plaid_ids = self._load_doc_ids_to_plaid_ids()
        plaid_ids_to_doc_ids = self._load_plaid_ids_to_doc_ids()
        
        # Get plaid IDs to delete
        plaid_ids_to_delete = []
        for doc_id in doc_ids:
            if doc_id in doc_ids_to_plaid_ids:
                plaid_ids_to_delete.append(doc_ids_to_plaid_ids[doc_id])
        
        if not plaid_ids_to_delete:
            doc_ids_to_plaid_ids.close()
            plaid_ids_to_doc_ids.close()
            return self
        
        # Delete from fast-plaid index
        self.fast_plaid.delete(subset=plaid_ids_to_delete)
        
        # Remove old mappings
        for doc_id in doc_ids:
            if doc_id in doc_ids_to_plaid_ids:
                plaid_id = doc_ids_to_plaid_ids[doc_id]
                del doc_ids_to_plaid_ids[doc_id]
                del plaid_ids_to_doc_ids[plaid_id]
        
        # Rebuild mappings since fast-plaid re-indexes after delete
        # Get remaining doc_ids in their original plaid_id order
        remaining_items = sorted(
            [(int(k), v) for k, v in plaid_ids_to_doc_ids.items()],
            key=lambda x: x[0]
        )
        
        # Clear and rebuild with new sequential IDs
        for k in list(doc_ids_to_plaid_ids.keys()):
            del doc_ids_to_plaid_ids[k]
        for k in list(plaid_ids_to_doc_ids.keys()):
            del plaid_ids_to_doc_ids[k]
        
        for new_plaid_id, (_, doc_id) in enumerate(remaining_items):
            doc_ids_to_plaid_ids[doc_id] = new_plaid_id
            plaid_ids_to_doc_ids[new_plaid_id] = doc_id
        
        doc_ids_to_plaid_ids.commit()
        doc_ids_to_plaid_ids.close()
        
        plaid_ids_to_doc_ids.commit()
        plaid_ids_to_doc_ids.close()
        
        logger.info(f"Removed {len(plaid_ids_to_delete)} documents from index.")
        return self