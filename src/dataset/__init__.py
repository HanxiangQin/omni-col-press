from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.dataset.encode_dataset import EncodeDataset
    from src.dataset.train_dataset import TrainDataset
    from src.dataset.util_dataset import MultiTrainDataset, DatasetSplitView
else:
    import sys
    from types import ModuleType

    _lazy_imports = {
        "EncodeDataset": ("src.dataset.encode_dataset", "EncodeDataset"),
        "TrainDataset": ("src.dataset.train_dataset", "TrainDataset"),
        "MultiTrainDataset": ("src.dataset.util_dataset", "MultiTrainDataset"),
        "DatasetSplitView": ("src.dataset.util_dataset", "DatasetSplitView"),
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
    "EncodeDataset",
    "TrainDataset",
    "MultiTrainDataset",
    "DatasetSplitView",
]
