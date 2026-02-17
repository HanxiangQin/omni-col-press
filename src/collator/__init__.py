from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.collator.base_collator import BaseMultiModalCollator
    from src.collator.qwenvl_collator import QwenVLCollator
    from src.collator.qwenomni_collator import QwenOmniCollator
else:
    import sys
    from types import ModuleType

    _lazy_imports = {
        "BaseMultiModalCollator": ("src.collator.base_collator", "BaseMultiModalCollator"),
        "QwenVLCollator": ("src.collator.qwenvl_collator", "QwenVLCollator"),
        "QwenOmniCollator": ("src.collator.qwenomni_collator", "QwenOmniCollator"),
    }

    def __getattr__(name: str):
        if name in _lazy_imports:
            module_path, class_name = _lazy_imports[name]
            module = __import__(module_path, fromlist=[class_name], level=0)
            value = getattr(module, class_name)
            setattr(sys.modules[__name__], name, value)
            return value
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    

from .audio_process import process_audio_info
from .vision_process import (
    extract_vision_info,
    fetch_image,
    fetch_video,
    process_vision_info,
    smart_resize,
)


def process_mm_info_local(conversations, use_audio_in_video, return_video_kwargs=False):
    audios = process_audio_info(conversations, use_audio_in_video)
    vision = process_vision_info(conversations, return_video_kwargs=return_video_kwargs)
    return (audios,) + vision

__all__ = [
    "BaseMultiModalCollator",
    "QwenVLCollator",
    "QwenOmniCollator",
]
