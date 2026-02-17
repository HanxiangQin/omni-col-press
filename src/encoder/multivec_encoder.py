import torch
from torch import Tensor
from src.encoder.base_encoder import EncoderModel
from src.utils import colbert_sim

import logging
logger = logging.getLogger(__name__)

class MultiVecEncoder(EncoderModel):
    def compute_similarity(self, q_reps, p_reps, q_mask=None, p_mask=None):
        if self.pooling in ['memory', 'colbert', 'hierarchical_clustering', 'resize', 'select']:
            return colbert_sim(query=q_reps, doc=p_reps, query_mask=q_mask, doc_mask=p_mask, normalize=False, return_argmax=False, use_full_precision=False)
        elif q_reps.dim() == 2 and p_reps.dim() == 2: # (B, D)
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        else:
            raise ValueError(f"Unsupported similarity computation for pooling method: {self.pooling}")

    def encode(self, inputs, is_query: bool = False):
        if inputs is None:
            return None
        
        attention_mask: torch.Tensor = inputs.get('attention_mask')
        outputs = self.model(**inputs, return_dict=True, output_hidden_states=True, use_cache=False)
        if isinstance(outputs, Tensor):
            hidden_states = [outputs]
        else:
            hidden_states = outputs.hidden_states
        embeddings, attention_mask = self._pooling(hidden_states, attention_mask, is_query=is_query)
        return embeddings, attention_mask

    def encode_passage(self, passage_inputs):
        return self.encode(passage_inputs, is_query=False)

    def encode_query(self, query_inputs):
        return self.encode(query_inputs, is_query=True)