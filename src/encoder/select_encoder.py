import torch
import torch.nn.functional as F
from typing import Optional
from transformers import PreTrainedModel
import copy
from src.encoder.multivec_encoder import MultiVecEncoder
from src.arguments import ModelArguments

import logging
logger = logging.getLogger(__name__)


class AttentionSelectEncoder(MultiVecEncoder):
    """
    Attention-Guided Clustering (AGC) Encoder with Learned Universal Query.

    This encoder implements the AGC method for representation compression. It uses 
    Learned Universal Query tokens to compute saliency scores over document tokens, 
    then selects the top-k most salient tokens and performs attention-guided clustering 
    to produce the final compressed representation.

    Key concepts:
    - Universal Query tokens (Ψ): Trainable tokens appended to document tokens that learn
      to identify and aggregate salient information across diverse documents
    - Saliency scores (α): Attention weights from Universal Query tokens to document tokens
      at the output layer, indicating the importance of each document token
    - AGC: Attention-Guided Clustering that uses the top-k salient tokens as cluster 
      centroids and pools nearby tokens to form the final representation

    This implementation computes attention at the OUTPUT LAYER ONLY, which allows using
    flash_attention_2 for the main forward pass (much faster and memory efficient).
    The Universal Query token embeddings at the final layer have already aggregated 
    information via bidirectional attention throughout the network.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        pooling: str = 'select',
        normalize: bool = True,
        temperature: float = 1.0,
        model_args: ModelArguments = None,
    ):
        super().__init__(model, pooling, normalize, temperature, model_args)
        
        # Number of Universal Query tokens appended to document tokens
        self.num_proxy_tokens = model_args.num_appending_token
        # Number of output representation vectors (cluster centroids); defaults to num_proxy_tokens
        self.num_select_tokens = model_args.num_repr_vectors or self.num_proxy_tokens
        
        # AGC (Attention-Guided Clustering) settings:
        # When enabled, selected tokens serve as cluster centroids and nearby tokens are pooled
        # cluster_centroid_weight: Weight for centroids in cluster averaging (1.0 = equal, >1.0 = more weight to centroids)
        self.use_cluster_pooling = model_args.use_cluster_pooling
        self.cluster_centroid_weight = model_args.cluster_centroid_weight
        self.use_attn_weight_cluster_pooling = model_args.use_attn_weight_cluster_pooling

        self._enable_eager_attention()

        logger.info(
            f"AGC Encoder initialized with {self.num_proxy_tokens} Universal Query tokens, "
            f"num_output_vectors={self.num_select_tokens}, "
            f"use_cluster_pooling={self.use_cluster_pooling}, "
            f"cluster_centroid_weight={self.cluster_centroid_weight}, "
            f"use_attn_weight_cluster_pooling={self.use_attn_weight_cluster_pooling}"
        )

    def encode(self, inputs, is_query: bool = False):
        if inputs is None:
            return None, None
        
        attention_mask: torch.Tensor = inputs.get('attention_mask')
        
        if is_query:
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            embeddings = hidden_states[-1]
            if self.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings, attention_mask
        else:
            outputs = self.model(**inputs, return_dict=True, output_hidden_states=True, output_attentions=True, use_cache=False)
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions
            # logger.info(f"Attention weights: {attentions[-1].shape, attentions[-1].dtype, attentions[-1].device}")
            result = self._encode_passage_with_selection(hidden_states[-1], attentions[-1], attention_mask)
            return result

    def _encode_passage_with_selection(
        self, 
        last_hidden_state: torch.Tensor,
        last_attention_weights: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        batch_size, seq_len, hidden_dim = last_hidden_state.shape
        
        doc_mask, query_mask = self._create_doc_query_masks(attention_mask, seq_len)  # Both: (batch_size, seq_len)
        saliency_scores = self._compute_saliency_from_universal_query(
            last_attention_weights, doc_mask, query_mask
        )  
        # (batch_size, seq_len) 
        # Saliency scores for document tokens computed from Universal Query attention
        # Length matches the full sequence (padded)
        
        # Select top-k salient tokens and apply AGC (Attention-Guided Clustering) if enabled
        embeddings, output_mask = self._select_and_cluster(last_hidden_state, saliency_scores, doc_mask, use_cluster_pooling=self.use_cluster_pooling, use_attn_weight_cluster_pooling=self.use_attn_weight_cluster_pooling)
        
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings, output_mask

    def _create_doc_query_masks(
        self,
        attention_mask: torch.Tensor,
        seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create separate masks for document tokens and Universal Query tokens.
        
        Returns:
            doc_mask: Mask indicating document token positions
            query_mask: Mask indicating Universal Query token positions
        """
        left_padding = attention_mask is None or (attention_mask[:, -1].sum() == attention_mask.shape[0])
        
        batch_size = attention_mask.shape[0]
        n_query = self.num_proxy_tokens  # Number of Universal Query tokens
        device = attention_mask.device
        dtype = attention_mask.dtype
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=dtype)
        doc_mask = attention_mask.clone()
        query_mask = attention_mask.clone()
        
        if left_padding:
            # Left padding: sequence is [padding, doc_tokens, universal_query_tokens]
            doc_mask[:, -n_query:] = 0
            query_mask[:, :-n_query] = 0
        else:
            # Right padding: sequence is [doc_tokens, universal_query_tokens, padding]
            actual_seq_len = attention_mask.sum(dim=-1)
            doc_len = actual_seq_len - n_query
            doc_mask[:, doc_len:] = 0
            query_mask[:, :doc_len] = 0
        
        return doc_mask, query_mask

    def _enable_eager_attention(self):
        config = None
        last_attention_layer = self._get_last_attention_layer()
        if hasattr(last_attention_layer, 'config'):
            config = last_attention_layer.config
        if config is None:
            raise RuntimeError("Could not find attention configuration in model structure")
        copied_config = copy.deepcopy(config)
        copied_config._attn_implementation = 'eager'
        last_attention_layer.config = copied_config

    def _get_last_attention_layer(self):
        """
        Navigate to the last decoder layer's attention module
        Expected Path: model.model.language_model.layers[-1].self_attn
        """
        # self (encoder) -> [model(PeftModel)] -> model (backbone) -> language model
        if hasattr(self.model, 'model'):
            backbone_model = self.model.model
        else:
            backbone_model = self.model

        if hasattr(backbone_model, 'language_model'):
            language_model = backbone_model.language_model
        elif hasattr(backbone_model, 'model'):
            language_model = backbone_model.model
        else:
            language_model = backbone_model

        if hasattr(language_model, 'layers'):
            last_layer = language_model.layers[-1]
            if hasattr(last_layer, 'self_attn'):
                return last_layer.self_attn
        
        raise RuntimeError("Could not find last attention layer in model structure")

    def _compute_saliency_from_universal_query(
        self,
        last_attention_weights: torch.Tensor,
        doc_mask: torch.Tensor,
        query_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute saliency scores for document tokens using attention from Universal Query tokens.
        
        The Universal Query tokens attend to document tokens, and their attention weights
        indicate the importance (saliency) of each document token. We average the attention
        weights across all Universal Query tokens and attention heads.
        
        Args:
            last_attention_weights: (batch, num_heads, seq_len, seq_len)
            doc_mask: (batch, seq_len) - mask for document token positions
            query_mask: (batch, seq_len) - mask for Universal Query token positions
            
        Returns:
            saliency_scores: (batch, seq_len) - saliency score for each position
        """
        batch_size, num_heads, seq_len, _ = last_attention_weights.shape
        # Average attention weights across heads
        attn_scores = last_attention_weights.mean(dim=1, keepdim=False)  # (batch, seq_len, seq_len)
        
        # Extract attention from Universal Query tokens to all positions
        query_attn_scores = attn_scores[query_mask.bool()].view(batch_size, -1, seq_len)  # (batch, num_query_tokens, seq_len)
        
        # Mask out non-document positions (only compute saliency for document tokens)
        neg_inf = torch.tensor(-float('inf'), dtype=query_attn_scores.dtype, device=query_attn_scores.device)
        query_attn_scores = query_attn_scores.masked_fill(~doc_mask.bool().unsqueeze(1), neg_inf)

        # Normalize attention over document tokens
        query_attn_scores = F.softmax(query_attn_scores, dim=-1)  # (batch, num_query_tokens, seq_len)
        
        # Average saliency across all Universal Query tokens
        saliency_scores = query_attn_scores.mean(dim=1, keepdim=False)  # (batch, seq_len)
        return saliency_scores

    def _select_and_cluster(
        self,
        doc_hidden_states: torch.Tensor,
        saliency_scores: torch.Tensor,
        doc_mask: torch.Tensor,
        use_cluster_pooling: bool = False,
        use_attn_weight_cluster_pooling: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k salient tokens and optionally apply Attention-Guided Clustering (AGC).
        
        First, select the top m document tokens based on saliency scores:
        C = TopKSelect(Z^(L)_X, α, m)
        
        If use_cluster_pooling is enabled (AGC), the selected tokens serve as cluster
        centroids and we pool all tokens within each cluster based on similarity.
        
        Args:
            doc_hidden_states: (batch, n_doc, hidden_dim) - document token embeddings
            saliency_scores: (batch, n_doc) - saliency score for each document token
            doc_mask: (batch, n_doc) - mask indicating valid document token positions
            use_cluster_pooling: Whether to apply AGC clustering
            use_attn_weight_cluster_pooling: Whether to use saliency scores as pooling weights
            
        Returns:
            output_embeddings: (batch, m, hidden_dim) - final representation vectors
            output_mask: (batch, m) - mask indicating valid output positions
        """
        batch_size, seq_len, hidden_dim = doc_hidden_states.shape
        m = self.num_select_tokens  # Number of output vectors
        
        # Select top-k most salient tokens as cluster centroids
        # Note: saliency_scores has -inf for padded positions (from _compute_saliency_from_universal_query),
        # so topk will naturally select valid document tokens first. If valid_doc_len < m, some padded
        # positions may be selected, but they will be masked out in the output.
        # topk_indices shape: (batch, m)
        _, topk_indices = torch.topk(saliency_scores, k=m, dim=-1, largest=True, sorted=False)
        
        # Create mask for selected tokens based on whether they came from valid doc positions
        # selected_doc_mask shape: (batch, m)
        selected_doc_mask = torch.gather(doc_mask, dim=1, index=topk_indices)
        
        # Gather the selected embeddings (cluster centroids)
        # Expand indices for gathering: (batch, m) -> (batch, m, hidden_dim)
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        centroid_embeddings = torch.gather(doc_hidden_states, dim=1, index=expanded_indices)
        
        if use_cluster_pooling:
            # Use selected tokens as centroids and pool nearby tokens
            # If use_attn_weight_cluster_pooling, use saliency scores as pooling weights (AGC)
            # Otherwise, use uniform weights (with optional centroid boosting)
            weights = saliency_scores if use_attn_weight_cluster_pooling else None
            output_embeddings = self._cluster_pool(
                doc_hidden_states, centroid_embeddings, topk_indices, doc_mask, weights
            )
        else:
            # Direct selection: return selected embeddings without clustering
            output_embeddings = centroid_embeddings

        return output_embeddings, selected_doc_mask

    def _cluster_pool(
        self,
        doc_hidden_states: torch.Tensor,
        centroids: torch.Tensor,
        centroid_indices: torch.Tensor,
        doc_mask: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cluster pooling: Assign each document token to its nearest centroid and compute 
        weighted average within each cluster.
        
        This implements cluster pooling where:
        1. Each token is assigned to the most similar centroid (hard assignment)
        2. Tokens within each cluster are pooled via weighted averaging
        3. Centroids can be given extra weight via cluster_centroid_weight
        
        When `weights` (saliency scores) are provided, this becomes Attention-Guided 
        Clustering (AGC) where tokens contribute proportionally to their saliency.
        When `weights` is None, uniform weights are used (with optional centroid boosting).
        
        Args:
            doc_hidden_states: (batch, seq_len, hidden_dim) - all document token embeddings
            centroids: (batch, m, hidden_dim) - selected centroid embeddings (most salient tokens)
            centroid_indices: (batch, m) - indices of centroid tokens in doc_hidden_states
            doc_mask: (batch, seq_len) - mask indicating valid document token positions
            weights: Optional[torch.Tensor] - saliency weights for each document token; 
                     if provided, enables attention-guided pooling; 
                     if None, uniform weights are used (optionally boosted for centroids)
            
        Returns:
            pooled_embeddings: (batch, m, hidden_dim) - cluster-pooled representations
        """
        batch_size, seq_len, hidden_dim = doc_hidden_states.shape
        _, m, _ = centroids.shape
        
        # Normalize embeddings for cosine similarity-based cluster assignment
        doc_norm = F.normalize(doc_hidden_states, p=2, dim=-1)  # (batch, seq_len, hidden_dim)
        centroid_norm = F.normalize(centroids, p=2, dim=-1)  # (batch, m, hidden_dim)
        
        # Compute similarity between all document tokens and cluster centroids
        # (batch, seq_len, hidden_dim) @ (batch, hidden_dim, m) -> (batch, seq_len, m)
        similarity = torch.bmm(doc_norm, centroid_norm.transpose(1, 2))
        
        # Hard cluster assignment: assign each token to its most similar centroid
        # Note: Invalid tokens may be assigned to any centroid, but they will have zero weight
        # in the aggregation due to doc_mask, so they won't contribute to the output.
        # assignments shape: (batch, seq_len)
        assignments = similarity.argmax(dim=-1)
        
        # Create weight tensor for pooling if not provided 
        # Use uniform weights or boosted centroid weights
        # (batch, seq_len)
        if weights is None:
            # Optionally apply extra weight to centroid tokens
            if self.cluster_centroid_weight != 1.0:
                # Create a mask for centroid positions
                centroid_mask = torch.zeros(batch_size, seq_len, device=doc_hidden_states.device, dtype=doc_hidden_states.dtype)
                centroid_mask.scatter_(1, centroid_indices, 1.0)
                
                # Apply extra weight to centroids: weight = 1 + (centroid_weight - 1) * is_centroid
                weight_factor = torch.tensor(self.cluster_centroid_weight - 1.0, dtype=doc_hidden_states.dtype, device=doc_hidden_states.device)
                weights = torch.ones(batch_size, seq_len, device=doc_hidden_states.device, dtype=doc_hidden_states.dtype)
                weights = weights + weight_factor * centroid_mask
            else:
                weights = torch.ones(batch_size, seq_len, device=doc_hidden_states.device, dtype=doc_hidden_states.dtype)
        
        # Apply doc_mask to zero out weights for non-document positions
        weights = weights * doc_mask.to(weights.dtype)
        
        # Create weighted assignment matrix: (batch, seq_len, m)
        one_hot = F.one_hot(assignments, num_classes=m).to(doc_hidden_states.dtype)
        weighted_one_hot = one_hot * weights.unsqueeze(-1)
        
        # Compute weighted sum of tokens within each cluster
        # (batch, m, seq_len) @ (batch, seq_len, hidden_dim) -> (batch, m, hidden_dim)
        cluster_sums = torch.bmm(weighted_one_hot.transpose(1, 2), doc_hidden_states)
        
        # Compute sum of weights per cluster for normalization: (batch, m)
        min_val = torch.tensor(1e-8, dtype=doc_hidden_states.dtype, device=doc_hidden_states.device)
        cluster_weight_sums = weighted_one_hot.sum(dim=1).clamp(min=min_val)  # Avoid division by zero
        # Compute weighted average to get final cluster representations
        pooled_embeddings = cluster_sums / cluster_weight_sums.unsqueeze(-1)
        
        return pooled_embeddings