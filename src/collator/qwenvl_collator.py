import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from qwen_vl_utils import process_vision_info

from .base_collator import BaseMultiModalCollator
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


logger = logging.getLogger(__name__)


@dataclass
class QwenVLCollator(BaseMultiModalCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_qwen3vl = isinstance(self.processor, Qwen3VLProcessor)

    def _create_message_template(self, items: List[Dict[str, Any]], is_query: bool) -> Any:
        patch_size = 16 if self.is_qwen3vl else 14
        spatial_pooling_factor = 2
        temporal_pooling_factor = 2
        patch_pixels = (patch_size * spatial_pooling_factor)**2
        
        messages = []
        for item in items:
            content = []
            if item.get("text", None) is not None:
                max_length = self.data_args.query_max_len if is_query else self.data_args.passage_max_len
                text = self.processor.tokenizer.decode(
                    self.processor.tokenizer.encode(item["text"], max_length=max_length, truncation=True)
                )
                content.append({'type': 'text', 'text': text})
            if item.get("image", None) is not None:
                content.append({
                    'type': 'image', 'image': item["image"], 
                    'max_pixels': 1280 * patch_pixels,
                    'min_pixels': 784 * patch_pixels,
                })
                # Cap image at 1280 tokens
            if item.get("video", None) is not None:
                content.append({
                    'type': 'video', 'video': item["video"], 
                    'nframes': 24,
                    'max_pixels': 54 * temporal_pooling_factor * patch_pixels,
                    'min_pixels': 48 * temporal_pooling_factor * patch_pixels
                })
                # Cap video at 24 * 54 = 1296 tokens
            
            message = [{'role': 'user', 'content': content}]
            messages.append(message)
        
        return messages
    
    def _preprocess_data(self, items: List[Dict[str, Any]], messages: List[List[Dict[str, Any]]], is_query: bool) -> Any:
        text_inputs = []
        image_inputs = []
        video_inputs = []
        video_metadatas = []
        kept_ids = []
        preprocessed_data = {}
        failed_indices = set()
        
        # First pass: process all items and track failures
        results = []  # Store (success, text_input, image_input, video_input, video_metadata, id)
        for idx, (item, msg) in enumerate(zip(items, messages)):
            text_input_item = image_input_item = video_input_item = None
            video_metadata_item = None
            video_kwargs_item = {}
            try:
                text_input_item = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                text_input_item = self._modify_text_input(text_input_item, is_query)
                if item.get('image', None) or item.get('video', None):
                    if self.is_qwen3vl:
                        image_input_item, video_input_item, video_kwargs_item = process_vision_info(
                            [msg], 
                            image_patch_size=16, 
                            return_video_kwargs=True, 
                            return_video_metadata=True
                        )
                        if video_input_item is not None:
                            video_input_item, video_metadata_item = zip(*video_input_item)
                            video_input_item, video_metadata_item = list(video_input_item), list(video_metadata_item)
                    else:
                        # Qwen2.5-VL: use default image_patch_size=14
                        image_input_item, video_input_item = process_vision_info([msg])
                results.append((True, text_input_item, image_input_item, video_input_item, video_metadata_item, item.get("id", None)))
            except Exception as e:
                self._handle_failed_item(item, msg, e)
                failed_indices.add(idx)
                results.append((False, None, None, None, None, None))
        
        # Second pass: replace failed items with random successful ones
        for idx, result in enumerate(results):
            if result[0]:  # Success
                _, text_input_item, image_input_item, video_input_item, video_metadata_item, item_id = result
            else:  # Failed - try to replace with a random successful item
                replacement_idx = self._get_random_replacement_index(idx, failed_indices, len(items))
                if replacement_idx is None:
                    # No successful items available, skip
                    continue
                _, text_input_item, image_input_item, video_input_item, video_metadata_item, item_id = results[replacement_idx]
            
            text_inputs.append(text_input_item)
            image_inputs.extend(image_input_item or [])
            video_inputs.extend(video_input_item or [])
            video_metadatas.extend(video_metadata_item or [])
            kept_ids.append(item_id)

        preprocessed_data['text'] = text_inputs
        if len(image_inputs) > 0:
            preprocessed_data['images'] = image_inputs
        if len(video_inputs) > 0:
            preprocessed_data['videos'] = video_inputs
        if len(video_metadatas) > 0:
            preprocessed_data['video_metadata'] = video_metadatas
        # for Qwen3-VL, we need to set the following kwargs to False
        preprocessed_data['do_resize'] = False
        preprocessed_data['do_sample_frames'] = False
        return kept_ids, preprocessed_data