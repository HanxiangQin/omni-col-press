from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.dataset_utils import build_record
    from src.utils.utils import colbert_sim, pad_and_concat, pad_and_stack
    from src.utils.appending_tokens import (
        get_appending_token_strings,
        add_appending_tokens_to_tokenizer,
        initialize_appending_token_embeddings,
    )
    from src.utils.loaders import load_index, load_qrels, load_dataset_labels, load_ground_truth
    from src.utils.eval_utils import (
        calculate_recall_at_k,
        calculate_mrr,
        calculate_ndcg_at_k,
        get_relevant_docs_list,
    )
    from src.utils.dist_utils import (
        init_distributed,
        finalize_distributed,
        get_gloo_group,
        dist_gather_tensor,
        dist_gather_tensor_with_grads,
        list_array_split_np_style,
    )
else:
    import sys
    from types import ModuleType

    _lazy_imports = {
        # dataset_utils
        "build_record": ("src.utils.dataset_utils", "build_record"),
        # utils
        "colbert_sim": ("src.utils.utils", "colbert_sim"),
        "pad_and_concat": ("src.utils.utils", "pad_and_concat"),
        "pad_and_stack": ("src.utils.utils", "pad_and_stack"),
        # appending_tokens
        "get_appending_token_strings": ("src.utils.appending_tokens", "get_appending_token_strings"),
        "add_appending_tokens_to_tokenizer": ("src.utils.appending_tokens", "add_appending_tokens_to_tokenizer"),
        "initialize_appending_token_embeddings": ("src.utils.appending_tokens", "initialize_appending_token_embeddings"),
        # loaders
        "load_index": ("src.utils.loaders", "load_index"),
        "load_qrels": ("src.utils.loaders", "load_qrels"),
        "load_dataset_labels": ("src.utils.loaders", "load_dataset_labels"),
        "load_ground_truth": ("src.utils.loaders", "load_ground_truth"),
        # eval_utils
        "calculate_recall_at_k": ("src.utils.eval_utils", "calculate_recall_at_k"),
        "calculate_mrr": ("src.utils.eval_utils", "calculate_mrr"),
        "calculate_ndcg_at_k": ("src.utils.eval_utils", "calculate_ndcg_at_k"),
        "get_relevant_docs_list": ("src.utils.eval_utils", "get_relevant_docs_list"),
        # dist_utils
        "init_distributed": ("src.utils.dist_utils", "init_distributed"),
        "finalize_distributed": ("src.utils.dist_utils", "finalize_distributed"),
        "get_gloo_group": ("src.utils.dist_utils", "get_gloo_group"),
        "dist_gather_tensor": ("src.utils.dist_utils", "dist_gather_tensor"),
        "dist_gather_tensor_with_grads": ("src.utils.dist_utils", "dist_gather_tensor_with_grads"),
        "list_array_split_np_style": ("src.utils.dist_utils", "list_array_split_np_style"),
    }

    def __getattr__(name: str):
        if name in _lazy_imports:
            module_path, attr_name = _lazy_imports[name]
            module = __import__(module_path, fromlist=[attr_name], level=0)
            value = getattr(module, attr_name)
            setattr(sys.modules[__name__], name, value)
            return value
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # dataset_utils
    "build_record",
    # utils
    "colbert_sim",
    "pad_and_concat",
    "pad_and_stack",
    # appending_tokens
    "get_appending_token_strings",
    "add_appending_tokens_to_tokenizer",
    "initialize_appending_token_embeddings",
    # loaders
    "load_index",
    "load_qrels",
    "load_dataset_labels",
    "load_ground_truth",
    # eval_utils
    "calculate_recall_at_k",
    "calculate_mrr",
    "calculate_ndcg_at_k",
    "get_relevant_docs_list",
    # dist_utils
    "init_distributed",
    "finalize_distributed",
    "get_gloo_group",
    "dist_gather_tensor",
    "dist_gather_tensor_with_grads",
    "list_array_split_np_style",
]
