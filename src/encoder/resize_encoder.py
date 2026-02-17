import torch
from torch import Tensor
from src.encoder.multivec_encoder import MultiVecEncoder
from transformers import PreTrainedModel
from src.arguments import ModelArguments
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

class SequenceResizerEncoder(MultiVecEncoder):
    def __init__(self,
                 model: PreTrainedModel,
                 pooling: str = 'resize',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 model_args: ModelArguments = None,
                 ):
        super().__init__(model, pooling, normalize, temperature, model_args)
        if model_args is None:
            raise ValueError("model_args must be provided for SequenceResizerEncoder.")

        self.resizer_input_size = model_args.resizer_input_size
        self.resizer_output_size = model_args.resizer_output_size
        self.resizer_hidden_size = model_args.resizer_hidden_size

        if self.resizer_input_size is None or self.resizer_output_size is None:
            raise ValueError("resizer_input_size and resizer_output_size must be set in ModelArguments for SequenceResizerEncoder.")
        if self.resizer_input_size <= 0 or self.resizer_output_size <= 0:
            raise ValueError("resizer_input_size and resizer_output_size must be positive integers.")

        # MLP that projects along the sequence dimension only.
        if self.resizer_hidden_size is not None and self.resizer_hidden_size > 0:
            self.sequence_resizer = nn.Sequential(
                nn.Linear(self.resizer_input_size, self.resizer_hidden_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.resizer_hidden_size, self.resizer_output_size, bias=True),
            )
        else:
            self.sequence_resizer = nn.Linear(self.resizer_input_size, self.resizer_output_size, bias=True)

    def encode(self, inputs, is_query: bool = False):
        if inputs is None:
            return None

        attention_mask: torch.Tensor = inputs.get('attention_mask')
        outputs = self.model(**inputs, return_dict=True, output_hidden_states=True, use_cache=False)
        if isinstance(outputs, Tensor):
            hidden_states = [outputs]
        else:
            hidden_states = outputs.hidden_states

        if is_query:
            embeddings = hidden_states[-1]
        else:
            # use last hidden state as token-level representations
            token_embeddings = hidden_states[-1]  # [B, seq_len, hidden_dim]
            token_embeddings, attention_mask = self._trim_or_pad(token_embeddings, attention_mask)

            # project the sequence length dimension to a fixed size while keeping hidden dim untouched
            seq_first = token_embeddings.transpose(1, 2)  # [B, hidden_dim, seq_len_fixed]
            resized = self.sequence_resizer(seq_first)  # [B, hidden_dim, resizer_output_size]
            embeddings = resized.transpose(1, 2)  # [B, resizer_output_size, hidden_dim]

            # we produce a dense multi-vector representation; mask is not needed
            attention_mask = None
            
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings, attention_mask

    def _trim_or_pad(self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None):
        """
        Trim from the left (keep the most recent tokens) and left-pad when shorter
        so the projection layer always receives resizer_input_size tokens.
        """
        batch_size, seq_len, hidden_dim = embeddings.shape
        target_len = self.resizer_input_size

        if seq_len > target_len:
            embeddings = embeddings[:, -target_len:, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -target_len:]
        elif seq_len < target_len:
            pad_len = target_len - seq_len
            pad_emb = embeddings.new_zeros((batch_size, pad_len, hidden_dim))
            embeddings = torch.cat([pad_emb, embeddings], dim=1)
            if attention_mask is not None:
                pad_mask = attention_mask.new_zeros((batch_size, pad_len))
                attention_mask = torch.cat([pad_mask, attention_mask], dim=1)

        return embeddings, attention_mask