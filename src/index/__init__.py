from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.index.base_index import MultiModalIndex
    from src.index.multivec_index import MultiVecIndex
    from src.index.fastplaid_index import FastPlaidIndex
else:
    import sys
    from types import ModuleType

    _lazy_imports = {
        "MultiModalIndex": ("src.index.base_index", "MultiModalIndex"),
        "MultiVecIndex": ("src.index.multivec_index", "MultiVecIndex"),
        "FastPlaidIndex": ("src.index.fastplaid_index", "FastPlaidIndex"),
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
    "MultiModalIndex",
    "MultiVecIndex",
    "FastPlaidIndex",
]
