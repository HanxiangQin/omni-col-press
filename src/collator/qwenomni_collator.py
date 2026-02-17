import logging
from typing import List, Dict, Any
from dataclasses import dataclass

# from qwen_omni_utils.v2_5 import process_mm_info
# from qwen_omni_utils import process_mm_info
from . import process_mm_info_local as process_mm_info
from . import BaseMultiModalCollator

logger = logging.getLogger(__name__)

@dataclass
class QwenOmniCollator(BaseMultiModalCollator):
    def _create_message_template(self, items: List[Dict[str, Any]], is_query: bool) -> Any:
        patch_size = 14
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
                    'max_pixels': 48 * temporal_pooling_factor * patch_pixels,
                    'min_pixels': 42 * temporal_pooling_factor * patch_pixels
                })
                # Cap video at 24 * 54 = 1296 tokens
            if item.get("audio", None) is not None:
                content.append({'type': 'audio', 'audio': item["audio"]})
            
            message = [{'role': 'user', 'content': content}]
            messages.append(message)
        
        return messages
    
    def _preprocess_data(self, items: List[Dict[str, Any]], messages: List[List[Dict[str, Any]]], is_query: bool) -> Any:
        text_inputs = []
        audio_inputs = []
        image_inputs = []
        video_inputs = []
        kept_ids = []
        failed_indices = set()
        
        # First pass: process all items and track failures
        results = []  # Store (success, text_input, audio_input, image_input, video_input, id)
        for idx, (item, msg) in enumerate(zip(items, messages)):
            text_input_item = audio_input_item = image_input_item = video_input_item = None
            try:
                text_input_item = self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                )
                text_input_item = self._modify_text_input(text_input_item, is_query)

                if item.get('image', None) or item.get('video', None) or item.get('audio', None):
                    audio_input_item, image_input_item, video_input_item = process_mm_info([msg], use_audio_in_video=False)
                results.append((True, text_input_item, audio_input_item, image_input_item, video_input_item, item.get("id", None)))
            except Exception as e:
                self._handle_failed_item(item, msg, e)
                failed_indices.add(idx)
                results.append((False, None, None, None, None, None))
        
        # Second pass: replace failed items with random successful ones
        for idx, result in enumerate(results):
            if result[0]:  # Success
                _, text_input_item, audio_input_item, image_input_item, video_input_item, item_id = result
            else:  # Failed - try to replace with a random successful item
                replacement_idx = self._get_random_replacement_index(idx, failed_indices, len(items))
                if replacement_idx is None:
                    # No successful items available, skip
                    continue
                _, text_input_item, audio_input_item, image_input_item, video_input_item, item_id = results[replacement_idx]
            
            text_inputs.append(text_input_item)
            audio_inputs.extend(audio_input_item or [])
            image_inputs.extend(image_input_item or [])
            video_inputs.extend(video_input_item or [])
            kept_ids.append(item_id)

        preprocessed_data = {
            'text': text_inputs,
        }
        if len(image_inputs) > 0:
            preprocessed_data['images'] = image_inputs
        if len(video_inputs) > 0:
            preprocessed_data['videos'] = video_inputs
            # For ColQwenQmni only
            # preprocessed_data['videos_kwargs'] = {
            #     'do_sample_frames': False,
            # }
        if len(audio_inputs) > 0:
            preprocessed_data['audio'] = audio_inputs
        return kept_ids, preprocessed_data