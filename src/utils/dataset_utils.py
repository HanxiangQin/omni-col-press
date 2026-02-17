import os
from typing import Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def build_record(entry_id, text, image, video, audio, modalities: Dict[str, bool], prefix: str = '', asset_path: str = None) -> Dict[str, Any]:
    record = {
        "id": entry_id,
    }

    if modalities.get('text', False) == False:
        text = None
    record['text'] = prefix + (text or '')

    if modalities.get('image', False) and image is not None:
        if isinstance(image, Image.Image):
            record['image'] = image
        else:
            if asset_path is not None:
                image_path = os.path.join(asset_path, image)
            else:
                image_path = image
            if not os.path.exists(image_path):
                logger.warning(f"Image file {image_path} does not exist.")
                image_path = None
            record['image'] = image_path

    if modalities.get('video', False) and video is not None:
        if asset_path is not None:
            video_path = os.path.join(asset_path, video)
        else:
            video_path = video
        if not os.path.exists(video_path):
            logger.warning(f"Video file {video_path} does not exist.")
            video_path = None
        record['video'] = video_path
        
    if modalities.get('audio', False) and audio is not None:
        if isinstance(audio, dict) and 'array' in audio:
            audio = audio['array']
        else:
            if asset_path is not None:
                audio_path = os.path.join(asset_path, audio)
            else:
                audio_path = audio
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file {audio_path} does not exist.")
                audio_path = None
        record['audio'] = audio_path

    return record
