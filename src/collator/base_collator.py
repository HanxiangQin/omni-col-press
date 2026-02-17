import hashlib
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
import torch
from transformers import PreTrainedTokenizer, ProcessorMixin

from src.arguments import DataArguments, ModelArguments
from src.utils import get_appending_token_strings

logger = logging.getLogger(__name__)


@dataclass
class BaseMultiModalCollator(ABC):
    """
    Abstract base class for multimodal collators.
    Provides common functionality while allowing model-specific implementations.
    """
    data_args: DataArguments
    model_args: ModelArguments
    processor: ProcessorMixin
    is_training: bool = False
    
    def __post_init__(self):
        self._failed_records: List[Dict[str, Any]] = []
        # Pre-compute appending token suffix for passages
        # These tokens serve as Universal Query tokens (for AGC) or memory tokens depending on the pooling method
        self._memory_token_suffix = self._build_memory_token_suffix()
    
    def _build_memory_token_suffix(self) -> str:
        """
        Build the token suffix to append to passages.
        
        For AGC (Attention-Guided Clustering): These are Universal Query tokens that learn
        to identify salient document content.
        For memory pooling: These are memory tokens that aggregate document information.
        """
        if self.model_args.num_appending_token <= 0:
            return ""
        
        if self.model_args.use_parametric_appending_tokens:
            # Use distinct learned tokens: <|mem0|><|mem1|>... (Universal Query or memory tokens)
            appending_tokens = get_appending_token_strings(self.model_args.num_appending_token)
            return "".join(appending_tokens)
        else:
            # Use repeated <|endoftext|> tokens
            return "<|endoftext|>" * self.model_args.num_appending_token

    def _modify_text_input(self, text_input: str, is_query: bool) -> str:
        if self.data_args.append_eos_token:
            text_input = text_input + '<|endoftext|>'
        if not is_query and self._memory_token_suffix:
            # Append tokens (Universal Query tokens for AGC, or memory tokens for memory pooling)
            text_input = text_input + self._memory_token_suffix
        return text_input
    
    def __call__(self, features):
        if self.is_training:
            query_features = [f["query"] for f in features]
            passage_features = []
            for f in features:
                passage_features.extend(f["passages"])
            query_ids, query_inputs = self._create_tensor_inputs(items=query_features, is_query=True)
            passage_ids, passage_inputs = self._create_tensor_inputs(items=passage_features, is_query=False)
            return {
                "query_inputs": query_inputs,
                "query_ids": query_ids,
                "passage_inputs": passage_inputs,
                "passage_ids": passage_ids,
                "passage_ids_tensor": self._tensorize_ids(passage_ids),
            }
        else:
            ids, inputs = self._create_tensor_inputs(items=features, is_query=self.data_args.encode_is_query)
            return {
                "ids": ids,
                "inputs": inputs,
            }
    
    def _create_tensor_inputs(self, items: List[Dict[str, Any]], is_query: bool) -> Any:
        messages = self._create_message_template(items, is_query)
        ids, preprocessed_data = self._preprocess_data(items, messages, is_query)
        processor_kwargs = {
            'return_tensors': "pt",
            'padding': "longest",
        }
        processor_kwargs.update(preprocessed_data)
        return ids, self.processor(**processor_kwargs)

    def _handle_failed_item(
        self,
        item: Dict[str, Any],
        msg: List[Dict[str, Any]],
        e: Exception,
        **kwargs: Any,
    ) -> None:
        record = {
            'item': item,
            'error': repr(e),
            **kwargs,
        }
        try:
            contents = msg[0].get('content', []) if isinstance(msg, list) and len(msg) > 0 else []
        except Exception as e:
            logger.exception("Failed to get contents from message to log error")

        video_paths = None
        audio_paths = None

        if item.get('video', False):
            try:
                video_paths = [c.get('video') for c in contents if isinstance(c, dict) and c.get('type') == 'video']
            except Exception:
                video_paths = "error getting video paths from message"
            record['videos'] = video_paths
        if item.get('audio', False):
            try:
                audio_paths = [c.get('audio') for c in contents if isinstance(c, dict) and c.get('type') == 'audio']
            except Exception:
                audio_paths = "error getting audio paths from message"
            record['audio'] = audio_paths
        logger.error(f"process mm/vision info failed; replacing with random item | id={item['id']} | videos={video_paths} | audio={audio_paths} | error={e}")
        self._failed_records.append(record)

    def _get_random_replacement_index(
        self,
        current_idx: int,
        failed_indices: set,
        total_items: int,
    ) -> Optional[int]:
        """Get a random index from successful items to use as replacement for a failed item."""
        available_indices = [i for i in range(total_items) if i not in failed_indices and i != current_idx]
        if not available_indices:
            return None
        return random.choice(available_indices)

    def get_failures(self) -> List[Dict[str, Any]]:
        return list(self._failed_records)
    
    def _tensorize_ids(self, ids: List[Any]) -> torch.Tensor:
        if ids is None or len(ids) == 0:
            return torch.empty(0, dtype=torch.long)
        encoded = [self._stable_hash_id(id_value) for id_value in ids]
        return torch.tensor(encoded, dtype=torch.long)

    @staticmethod
    def _stable_hash_id(value: Any) -> int:
        if value is None:
            # if id is None, use a random integer to avoid being treated as positive target
            return random.randint(0, 1000000000)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            data = value
        else:
            data = str(value)
        digest = hashlib.sha1(data.encode("utf-8")).hexdigest()
        hashed = int(digest[:16], 16)  # take first 64 bits
        limit = (1 << 63) - 1
        return hashed % limit

    @abstractmethod
    def _create_message_template(self, items: List[Dict[str, Any]], is_query: bool) -> Any:
        pass

    @abstractmethod
    def _preprocess_data(self, items: List[Dict[str, Any]], messages: List[List[Dict[str, Any]]], is_query: bool) -> Tuple[Dict[str, Any], List[int]]:
        pass