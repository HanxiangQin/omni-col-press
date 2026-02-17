"""Utility functions for parametric appending tokens in compressed multivec pooling."""

import logging
from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.getLogger(__name__)

APPENDING_TOKEN_FORMAT = "<|mem{i}|>"

def get_appending_token_strings(num_tokens: int) -> List[str]:
    return [APPENDING_TOKEN_FORMAT.format(i=i) for i in range(num_tokens)]

def add_appending_tokens_to_tokenizer(
    processor: ProcessorMixin,
    num_tokens: int,
) -> List[str]:
    appending_tokens = get_appending_token_strings(num_tokens)
    
    # Add as special tokens so they won't be split
    num_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": appending_tokens}
    )
    
    logger.info(f"Added {num_added} appending tokens to tokenizer: {appending_tokens}")
    return appending_tokens

def initialize_appending_token_embeddings(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    appending_tokens: List[str],
    init_std: float = 0.02,
) -> List[int]:
    # Get token IDs
    appending_token_ids = tokenizer.convert_tokens_to_ids(appending_tokens)
    
    # Get the embedding layer
    # Handle different model architectures
    if hasattr(model, 'get_input_embeddings'):
        embed_layer = model.get_input_embeddings()
    elif hasattr(model, 'model') and hasattr(model.model, 'get_input_embeddings'):
        embed_layer = model.model.get_input_embeddings()
    else:
        raise ValueError("Cannot find input embeddings in model")
    
    embed_weight = embed_layer.weight
    hidden_size = embed_weight.shape[1]
    
    with torch.no_grad():
        for i, tok_id in enumerate(appending_token_ids):
            # Random normal initialization
            torch.nn.init.normal_(embed_weight[tok_id], mean=0.0, std=init_std)
            # Add index-based offset to ensure diversity
            # Use a simple linear offset across the hidden dimension
            offset = torch.linspace(-0.01, 0.01, hidden_size, device=embed_weight.device, dtype=embed_weight.dtype)
            embed_weight[tok_id] += offset * (i + 1)
    
    logger.info(
        f"Initialized {len(appending_token_ids)} appending token embeddings "
        f"(hidden_size={hidden_size}, init_std={init_std})"
    )
    
    return appending_token_ids
