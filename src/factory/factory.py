#!/usr/bin/env python3

import os
import random
import logging
from dataclasses import replace
from typing import Any, Dict, Tuple, Union, Optional, List

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoProcessor
from transformers.trainer_utils import IntervalStrategy

from src.arguments import ModelArguments, DataArguments, TrainingArguments, EvaluateArguments, IndexArguments
from src.encoder import EncoderModel, MultiVecEncoder, SequenceResizerEncoder, AttentionSelectEncoder
from src.collator import BaseMultiModalCollator, QwenVLCollator, QwenOmniCollator
from src.dataset import EncodeDataset, TrainDataset, MultiTrainDataset, DatasetSplitView
from src.trainer.retrieval_trainer import RetrievalTrainer
from src.utils import add_appending_tokens_to_tokenizer, initialize_appending_token_embeddings, get_appending_token_strings



logger = logging.getLogger(__name__)

def infer_torch_dtype(model_args: ModelArguments, training_args: Optional[TrainingArguments] = None) -> torch.dtype:
    if training_args is not None:
        if training_args.fp16:
            torch_dtype = torch.float16
        elif training_args.bf16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32
        return torch_dtype

    if model_args.dtype in ["float16", "fp16"]:
        torch_dtype = torch.float16
    elif model_args.dtype in ["bfloat16", "bf16"]:
        torch_dtype = torch.bfloat16
    elif model_args.dtype in ["float32", "fp32"]:
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {model_args.dtype}")
    return torch_dtype

def _prepare_model_parallel_kwargs() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    try:
        num_gpus = torch.cuda.device_count()
    except Exception:
        num_gpus = 0
    if num_gpus <= 1:
        return {}

    default_per_gpu_gib = int(os.environ.get("VIDREPR_MP_PER_GPU_GIB", 22))
    max_memory: Dict[Any, str] = {i: f"{default_per_gpu_gib}GiB" for i in range(num_gpus)}

    mp_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        "max_memory": max_memory,
        "torch_dtype": torch.float32,
        "low_cpu_mem_usage": True,
    }
    return mp_kwargs

def infer_inputs_device(model: torch.nn.Module) -> str:
    try:
        return str(model.get_input_embeddings().weight.device)
    except Exception:
        return str(next(model.parameters()).device)

def _detect_model_family(model_name_or_path: str) -> str:
    name_l = model_name_or_path.lower()
    if ("qwen2.5-vl" in name_l) or ("qwen3-vl" in name_l) or ("moca-qwen25vl" in name_l):
        return "qwen_vl"
    if (
        ("qwen2.5-omni" in name_l)
        or ("qwen3-omni" in name_l)
        or ("qwen3_omni" in name_l)
        or ("omniembed" in name_l)
        or ("colqwen-omni" in name_l)
    ):
        return "qwen_omni"
    raise ValueError(f"Model {model_name_or_path} not supported")

def _detect_transformer_cls(model_name_or_path: str):
    name_l = model_name_or_path.lower()
    if ("qwen2.5-vl" in name_l) or ("moca-qwen25vl" in name_l):
        from src.models.qwen2_5_vl_embed.qwen2_5_vl_embed import Qwen2_5ForEmbedding
        return Qwen2_5ForEmbedding
    elif ("qwen3-vl" in name_l):
        from src.models.qwen3_vl_embed.qwen3_vl_embed import Qwen3ForEmbedding
        return Qwen3ForEmbedding
    elif ("qwen3-omni" in name_l) or ("qwen3_omni" in name_l):
        from src.models.qwen3_omni_moe_embed.qwen3_omni_moe_embed import Qwen3OmniMoeForEmbedding
        return Qwen3OmniMoeForEmbedding
    elif ("qwen2.5-omni" in name_l):
        from src.models.qwen2_5_omni_embed.qwen2_5_omni_embed import Qwen2_5OmniForEmbedding
        return Qwen2_5OmniForEmbedding
    elif ("omniembed" in name_l):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        return Qwen2_5OmniThinkerForConditionalGeneration
    elif ("colqwen-omni" in name_l):
        from src.models.colqwen_omni.modeling_colqwen_omni import ColQwen2_5Omni
        return ColQwen2_5Omni
    raise ValueError(f"Model {model_name_or_path} not supported")

def _select_encoder_cls(model_args: ModelArguments):
    if model_args.pooling in ["resize"]:
        return SequenceResizerEncoder
    if model_args.pooling in ["select"]:
        return AttentionSelectEncoder
    return MultiVecEncoder

def load_encoder_model(model_args: ModelArguments, device: str, torch_dtype: torch.dtype, tokenizer_len: int = 0) -> Tuple[EncoderModel, bool]:
    if model_args.model_parallel:
        mp_kwargs = _prepare_model_parallel_kwargs()
        using_model_parallel = len(mp_kwargs) > 0
    else:
        mp_kwargs = {}
        using_model_parallel = False

    if using_model_parallel:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model_kwargs = {
        "attn_implementation": model_args.attn_implementation,
        "dtype": torch_dtype,
    }
    if using_model_parallel:
        model_kwargs.update(mp_kwargs)

    transformer_cls = _detect_transformer_cls(model_args.model_name_or_path)
    encoder_cls = _select_encoder_cls(model_args)
    logger.info(f"Loading model: {transformer_cls.__name__}, encoder_cls: {encoder_cls.__name__}, model_kwargs: {model_kwargs}")
    model = encoder_cls.load(transformer_cls, model_args, tokenizer_len=tokenizer_len, **model_kwargs)

    if not using_model_parallel:
        model.to(device)

    return model, using_model_parallel

def build_encoder_model(model_args: ModelArguments, train_args: TrainingArguments, torch_dtype: torch.dtype, tokenizer_len: int = 0) -> EncoderModel:
    transformer_cls = _detect_transformer_cls(model_args.model_name_or_path)
    encoder_cls = _select_encoder_cls(model_args)
    model = encoder_cls.build(
        transformer_cls,
        model_args,
        train_args,
        tokenizer_len=tokenizer_len,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
    )
    return model

def _initialize_appending_token_embeddings(model: EncoderModel, processor: AutoProcessor, num_tokens: int, tokenizer_len: int):
    # Initialize appending token embeddings after model is built
    if tokenizer_len > 0:
        appending_tokens = get_appending_token_strings(num_tokens)
        # Get the base model (unwrap LoRA if present)
        # The model structure is: EncoderModel -> model (PeftModel or base) -> base_model
        base_model = model.model
        # Unwrap PeftModel if present
        if hasattr(base_model, 'get_base_model'):
            base_model = base_model.get_base_model()
        elif hasattr(base_model, 'base_model'):
            base_model = base_model.base_model
        initialize_appending_token_embeddings(
            base_model, processor.tokenizer, appending_tokens
        )
        logger.info(f"Set up learned appending tokens (Universal Query / Memory Tokens): {appending_tokens}")

def _resolve_processor_id(model_args: ModelArguments) -> str:
    if model_args.processor_name_or_path:
        return model_args.processor_name_or_path
    if model_args.tokenizer_name:
        return model_args.tokenizer_name
    name_l = model_args.model_name_or_path.lower()
    if "omniembed" in name_l:
        return "Qwen/Qwen2.5-Omni-7B"
    elif "moca-qwen25vl-7b" in name_l:
        return "Qwen/Qwen2.5-VL-7B-Instruct"

    return model_args.model_name_or_path

def load_processor(model_args: ModelArguments) -> AutoProcessor:
    processor_id = _resolve_processor_id(model_args)
    processor_cls = AutoProcessor
    # for ColQwenQmni only
    # if "vidore/colqwen-omni-v0.1" in processor_id:
    #     from colpali_engine.models.qwen_omni.colqwen_omni.processing_colqwen_omni import ColQwen2_5OmniProcessor
    #     processor_cls = ColQwen2_5OmniProcessor
    processor = processor_cls.from_pretrained(
        processor_id,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        use_fast=True
    )
    processor.tokenizer.padding_side = "left"
    return processor

def create_collator(data_args: DataArguments, model_args: ModelArguments, processor: AutoProcessor, is_training: bool = False) -> BaseMultiModalCollator:
    family = _detect_model_family(model_args.model_name_or_path)
    if family == "qwen_vl":
        return QwenVLCollator(data_args, model_args, processor, is_training)
    if family == "qwen_omni":
        return QwenOmniCollator(data_args, model_args, processor, is_training)
    raise ValueError(f"Model {model_args.model_name_or_path} not supported")

def create_inference_components(
    device: str,
    model_args: ModelArguments,
    data_args: DataArguments,
    batch_size: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[
    EncoderModel,
    AutoProcessor,
    BaseMultiModalCollator,
    EncodeDataset,
    DataLoader,
    str,
    torch.dtype,
]:
    torch_dtype = infer_torch_dtype(model_args)
    processor = load_processor(model_args)
    if (model_args.use_parametric_appending_tokens and model_args.num_appending_token > 0):
        # Add tokens to tokenizer
        appending_tokens = add_appending_tokens_to_tokenizer(processor, model_args.num_appending_token)
    tokenizer_len = len(processor.tokenizer)
    
    model, using_model_parallel = load_encoder_model(model_args, device, torch_dtype, tokenizer_len=tokenizer_len)

    collator = create_collator(data_args, model_args, processor, is_training=False)
    device_for_inputs = infer_inputs_device(model) if using_model_parallel else device
    dataset = EncodeDataset(data_args)
    sampler = None
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=data_args.num_proc,
        shuffle=False,
    )
    logger.info(f"Loaded Model: {model.__class__.__name__}, Processor: {processor.__class__.__name__}, Collator: {collator.__class__.__name__}")
    logger.info(f"Using model parallel: {using_model_parallel}, device_for_inputs: {device_for_inputs}, torch_dtype: {torch_dtype}")
    return model, processor, collator, dataset, dataloader, device_for_inputs, torch_dtype

def _create_train_dataset_from_entry(
    data_args: DataArguments,
    entry: Dict[str, Any],
    trainer=None
) -> TrainDataset:
    """
    Create a TrainDataset from a YAML entry.
    
    Args:
        data_args: DataArguments with default settings
        entry: Dictionary from YAML with keys like:
            - train_path: path to training data (or name for unsplit)
            - valid_path: path to validation data (optional, for split datasets)
            - corpus_path: path to corpus
            - corpus_assets_path: path to assets
            - dataset_name: dataset name (e.g., 'json')
            - corpus_name: corpus name (e.g., 'json')
        trainer: Optional trainer
    """
    import os
    
    # Get dataset path
    ds_path = entry.get('train_path') or entry.get('name')
    if ds_path is None:
        raise ValueError(f"Dataset entry must have 'train_path' or 'name': {entry}")
    
    # Determine dataset type
    dataset_name = None
    ds_file = None
    if os.path.isdir(ds_path):
        dataset_name = ds_path
    elif entry.get('dataset_name'):
        dataset_name = entry['dataset_name']
        ds_file = ds_path
    elif ds_path.endswith('.jsonl'):
        dataset_name = 'json'
        ds_file = ds_path
    else:
        dataset_name = ds_path
    
    # Get corpus path
    corpus_path = entry.get('corpus_path')
    corpus_assets_path = entry.get('corpus_assets_path')
    
    # Determine corpus type
    corpus_name = None
    corpus_file = None
    if corpus_path is None:
        corpus_name = None
    elif os.path.isdir(corpus_path):
        corpus_name = corpus_path
    elif entry.get('corpus_name'):
        corpus_name = entry['corpus_name']
        corpus_file = corpus_path
    elif corpus_path.endswith('.jsonl'):
        corpus_name = 'json'
        corpus_file = corpus_path
    else:
        corpus_name = corpus_path

    logger.info(f"dataset_name: {dataset_name}, corpus_name: {corpus_name}, dataset_path: {ds_file}, corpus_path: {corpus_file}, corpus_assets_path: {corpus_assets_path}")
    
    return TrainDataset(
        data_args, trainer,
        dataset_name, corpus_name,
        dataset_path=ds_file,
        corpus_path=corpus_file,
        corpus_assets_path=corpus_assets_path
    )

def _create_datasets_from_yaml(
    data_args: DataArguments,
    train_yaml: Dict[str, Any],
    trainer=None
) -> Tuple[MultiTrainDataset, Optional[MultiTrainDataset]]:
    """
    Create train and validation MultiTrainDataset from YAML configuration.
    
    Args:
        data_args: DataArguments with default settings
        train_yaml: Parsed YAML dictionary with 'datasets' key
        trainer: Optional trainer
    
    Returns:
        Tuple of (train_dataset, valid_dataset). valid_dataset is None if no pre-split validation data.
    """
    datasets_list = train_yaml.get("datasets")
    if datasets_list is None:
        raise ValueError("train_yaml must contain 'datasets' entry")
    
    train_datasets = []
    valid_datasets = []
    
    for entry in datasets_list:
        is_split = entry.get('is_split', False)
        
        if is_split:
            # Pre-split dataset: create both train and valid
            train_path = entry.get('train_path')
            valid_path = entry.get('valid_path') or entry.get('val_path') or entry.get('valid')
            
            if train_path is None:
                raise ValueError(f"Pre-split dataset entry must have 'train_path': {entry}")
            if valid_path is None:
                raise ValueError(f"Pre-split dataset entry must have 'valid_path', 'val_path', or 'valid': {entry}")
            
            # Create train dataset
            train_entry = {**entry, 'train_path': train_path}
            train_datasets.append(_create_train_dataset_from_entry(data_args, train_entry, trainer))
            
            # Create validation dataset
            valid_entry = {**entry, 'train_path': valid_path}  # Reuse same structure
            valid_datasets.append(_create_train_dataset_from_entry(data_args, valid_entry, trainer))
        else:
            # Unsplit dataset: only create train (validation will be split later)
            dataset = _create_train_dataset_from_entry(data_args, entry, trainer)
            train_dataset, valid_dataset = _split_dataset_for_validation(
                dataset, data_args.validation_split_ratio, data_args.validation_split_seed
            )
            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)
    
    train_multi_dataset = MultiTrainDataset(train_datasets, trainer)
    valid_multi_dataset = MultiTrainDataset(valid_datasets, trainer) if valid_datasets else None
    
    return train_multi_dataset, valid_multi_dataset

def _build_val_data_args(data_args: DataArguments) -> DataArguments:
    return replace(
        data_args,
        dataset_name=data_args.val_dataset_name,
        dataset_config=data_args.val_dataset_config or data_args.dataset_config,
        dataset_path=data_args.val_dataset_path,
        dataset_split=data_args.val_dataset_split,
        corpus_name=data_args.val_corpus_name,
        corpus_config=data_args.val_corpus_config or data_args.corpus_config,
        corpus_path=data_args.val_corpus_path,
        corpus_split=data_args.val_corpus_split,
        assets_path=data_args.val_assets_path,
    )

def _split_dataset_for_validation(
    dataset: Union[TrainDataset, MultiTrainDataset],
    val_ratio: float,
    seed: int,
) -> Tuple[Union[TrainDataset, MultiTrainDataset, DatasetSplitView], Optional[DatasetSplitView]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"validation_split_ratio must be between 0 and 1, got {val_ratio}")

    total_examples = len(dataset)
    if total_examples < 2:
        logger.warning("Validation split skipped: dataset has fewer than 2 samples.")
        return dataset, None

    val_size = max(1, int(round(total_examples * val_ratio)))
    if val_size >= total_examples:
        val_size = total_examples - 1
    if val_size <= 0:
        logger.warning("Validation split skipped: computed validation size is zero.")
        return dataset, None

    indices = list(range(total_examples))
    random.Random(seed).shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        logger.warning("Validation split skipped: no samples left for training after split.")
        return dataset, None

    logger.info(
        "Created automatic validation split: %d training samples, %d validation samples.",
        len(train_indices),
        len(val_indices),
    )
    return DatasetSplitView(dataset, train_indices), DatasetSplitView(dataset, val_indices)


def _ensure_eval_strategy(training_args: TrainingArguments):
    if training_args.eval_strategy is None or training_args.eval_strategy == IntervalStrategy.NO:
        training_args.eval_strategy = IntervalStrategy.STEPS
        training_args.eval_steps = 100
        logger.info(
            "Evaluation strategy was 'no' but validation data is available. "
            "Switching to %s every %s steps.",
            training_args.eval_strategy,
            training_args.eval_steps,
        )

def create_train_components(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    index_args: IndexArguments,
) -> Tuple[
    EncoderModel,
    AutoProcessor,
    BaseMultiModalCollator,
    Union[TrainDataset, MultiTrainDataset, DatasetSplitView],
    RetrievalTrainer,
    torch.dtype,
]:
    processor = load_processor(model_args)
    tokenizer_len = 0
    if (model_args.use_parametric_appending_tokens and model_args.num_appending_token > 0):
        appending_tokens = add_appending_tokens_to_tokenizer(processor, model_args.num_appending_token)
        tokenizer_len = len(processor.tokenizer)
    
    torch_dtype = infer_torch_dtype(model_args, training_args)
    model = build_encoder_model(model_args, training_args, torch_dtype, tokenizer_len=tokenizer_len)
    _initialize_appending_token_embeddings(model, processor, model_args.num_appending_token, tokenizer_len)

    train_dataset: Union[TrainDataset, MultiTrainDataset]
    if data_args.train_yaml is not None:
        with open(data_args.train_yaml, "r", encoding="utf-8") as f:
            train_yaml = yaml.safe_load(f) or {}
        train_dataset, eval_dataset = _create_datasets_from_yaml(data_args, train_yaml)
        if eval_dataset is not None:
            logger.info("Using pre-split validation datasets from train_yaml configuration.")
    else:
        train_dataset = TrainDataset(data_args)
        eval_dataset = None

    collator = create_collator(data_args, model_args, processor, is_training=True)

    # Priority 1: Pre-split validation datasets already handled above
    # Priority 2: Use explicit validation dataset if provided
    if eval_dataset is None and data_args.val_dataset_name is not None:
        eval_data_args = _build_val_data_args(data_args)
        eval_dataset = TrainDataset(eval_data_args)
        logger.info("Using explicit validation dataset with split '%s'.", eval_data_args.dataset_split)
    # Priority 3: Split unsplit datasets automatically
    elif eval_dataset is None and data_args.validation_split_ratio:
        train_dataset, eval_dataset = _split_dataset_for_validation(
            train_dataset, data_args.validation_split_ratio, data_args.validation_split_seed
        )

    if not training_args.do_eval:
        logger.info("Evaluation is disabled, setting eval_dataset to None")
        eval_dataset = None
    if eval_dataset is not None:
        _ensure_eval_strategy(training_args)

    logger.info(f"train_dataset: {len(train_dataset)}, eval_dataset: {len(eval_dataset) if eval_dataset is not None else 'None'}")

    trainer = RetrievalTrainer(
        index_args=index_args,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    train_dataset.set_trainer(trainer)
    if eval_dataset is not None:
        eval_dataset.set_trainer(trainer)

    return model, processor, collator, train_dataset, trainer, torch_dtype