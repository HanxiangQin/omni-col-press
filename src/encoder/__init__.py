from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.encoder.base_encoder import EncoderModel, EncoderOutput
    from src.encoder.multivec_encoder import MultiVecEncoder
    from src.encoder.resize_encoder import SequenceResizerEncoder
    from src.encoder.select_encoder import AttentionSelectEncoder
else:
    import sys
    from types import ModuleType

    _lazy_imports = {
        "EncoderModel": ("src.encoder.base_encoder", "EncoderModel"),
        "EncoderOutput": ("src.encoder.base_encoder", "EncoderOutput"),
        "MultiVecEncoder": ("src.encoder.multivec_encoder", "MultiVecEncoder"),
        "SequenceResizerEncoder": ("src.encoder.resize_encoder", "SequenceResizerEncoder"),
        "AttentionSelectEncoder": ("src.encoder.select_encoder", "AttentionSelectEncoder"),
    }

    def __getattr__(name: str):
        if name in _lazy_imports:
            module_path, class_name = _lazy_imports[name]
            module = __import__(module_path, fromlist=[class_name], level=0)
            value = getattr(module, class_name)
            setattr(sys.modules[__name__], name, value)
            return value
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EncoderModel",
    "EncoderOutput",
    "MultiVecEncoder",
    "SequenceResizerEncoder",
    "AttentionSelectEncoder",
]
