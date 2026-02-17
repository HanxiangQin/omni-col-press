from dataclasses import dataclass, field
from typing import Optional, List, Dict
from transformers import TrainingArguments
import json

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    processor_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained processor or processor identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # for lora
    lora: bool = field(default=False,
        metadata={"help": "do parameter-efficient fine-tuning with lora"}
    )
    lora_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained lora model or model identifier from huggingface.co/models"}
    )
    extra_state_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to extra encoder state (e.g., SequenceResizer) safetensors; overrides auto-discovery near model/lora paths."}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "lora r"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "lora alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "lora dropout"}
    )
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "lora target modules"}
    )

    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={
            "help": 'The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)).'
        },
    )

    model_parallel: bool = field(
        default=False,
        metadata={"help": "whether to use model parallel"}
    )

    # Pooling method arguments

    pooling: str = field(
        default='cls',
        metadata={"help": "pooling method for query and passage encoder"}
    )

    num_repr_vectors: int = field(
        default=0,
        metadata={"help": "number of vectors to retain"}
    )

    normalize: bool = field(
        default=True,
        metadata={"help": "normalize query and passage representations"}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for logits"}
    )

    # Appending token arguments (for AGC and Memory Tokens)

    num_appending_token: int = field(
        default=0,
        metadata={"help": "number of appending tokens"}
    )
    use_parametric_appending_tokens: bool = field(
        default=False,
        metadata={"help": "use learned appending tokens (Universal Query for AGC, or Memory Tokens) instead of repeating <|endoftext|>"}
    )

    # Sequence Resizing arguments

    resizer_input_size: Optional[int] = field(
        default=None,
        metadata={"help": "Fixed input seq length fed to SequenceResizerEncoder projection"}
    )
    resizer_output_size: Optional[int] = field(
        default=None,
        metadata={"help": "Output vector count (seq dim) for SequenceResizerEncoder"}
    )
    resizer_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden size of SequenceResizerEncoder MLP; if None/<=0 use single linear"}
    )

    # AGC (Attention-Guided Clustering) encoder arguments
    use_cluster_pooling: bool = field(
        default=True,
        metadata={"help": "Enable AGC: use selected tokens as cluster centroids and pool nearby tokens"}
    )
    cluster_centroid_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for centroids in AGC pooling (1.0 = equal weight, >1.0 = more weight to centroids)"}
    )
    use_attn_weight_cluster_pooling: bool = field(
        default=True,
        metadata={"help": "Use saliency scores to weight the AGC cluster pooling"}
    )

    # Hierarchical Clustering arguments

    num_protected_tokens: int = field(
        default=1,
        metadata={"help": "number of protected tokens for hierarchical clustering pooling"}
    )
    protected_tokens_position: str = field(
        default="first",
        metadata={"help": "which end to protect tokens from: 'first' protects the first N tokens, 'last' protects the last N tokens"}
    )

    def get_num_protected_tokens(self) -> int:
        if self.num_repr_vectors == 1:
            return 0
        elif self.num_repr_vectors > self.num_protected_tokens:
            return self.num_protected_tokens
        else:
            return self.num_repr_vectors - 1

    def get_num_clusters(self) -> int:
        return self.num_repr_vectors - self.get_num_protected_tokens()

@dataclass
class DataArguments:
    dataset_name: str = field(
        default='json', metadata={"help": "huggingface dataset name"}
    )

    dataset_config: str = field(
        default=None, metadata={"help": "huggingface dataset config, useful for datasets with sub-datasets"}
    )

    dataset_path: str = field(
        default=None, metadata={"help": "Path to local data files or directory"}
    )

    dataset_split: str = field(
        default='train', metadata={"help": "dataset split"}
    )

    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    corpus_name: str = field(
        default=None, metadata={"help": "huggingface dataset name for corpus"}
    )

    corpus_config: str = field(
        default=None, metadata={"help": "huggingface dataset config for corpus, useful for datasets with sub-datasets"}
    )

    corpus_path: str = field(
        default=None, metadata={"help": "Path to local corpus files or directory"}
    )

    corpus_split: str = field(
        default='train', metadata={"help": "corpus split"}
    )

    train_yaml: str = field(
        default=None, metadata={"help": "yaml file for training datasets, if there is more multiple datasets used for training"}
    )

    assets_path: str = field(
        default=None, metadata={"help": "path to assets for corpus"}
    )

    val_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "huggingface dataset name for validation/dev data; defaults to dataset_name"}
    )
    val_dataset_config: Optional[str] = field(
        default=None, metadata={"help": "huggingface dataset config for validation/dev data; defaults to dataset_config"}
    )
    val_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to local validation/dev data files or directory"}
    )
    val_dataset_split: Optional[str] = field(
        default="train", metadata={"help": "dataset split to use for validation/dev data"}
    )
    val_corpus_name: Optional[str] = field(
        default=None, metadata={"help": "huggingface dataset name for validation/dev corpus; defaults to corpus_name"}
    )
    val_corpus_config: Optional[str] = field(
        default=None, metadata={"help": "huggingface dataset config for validation/dev corpus; defaults to corpus_config"}
    )
    val_corpus_path: Optional[str] = field(
        default=None, metadata={"help": "Path to local validation/dev corpus files or directory"}
    )
    val_corpus_split: Optional[str] = field(
        default="train", metadata={"help": "corpus split to use for validation/dev data"}
    )
    val_assets_path: Optional[str] = field(
        default=None, metadata={"help": "path to assets for validation/dev corpus"}
    )
    validation_split_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "If no explicit validation dataset is provided, reserve this fraction (0-1) of the training data for validation."
        },
    )
    validation_split_seed: int = field(
        default=42,
        metadata={"help": "Random seed to use when creating an automatic validation split."},
    )

    query_path: str = field(
        default=None, metadata={"help": "Path to the query"}
    )
    
    qrels_path: str = field(
        default=None, metadata={"help": "path to qrels file"}
    )

    use_dataset_labels: bool = field(
        default=False, metadata={"help": "use dataset labels to build ground truth"}
    )

    dataset_number_of_shards: int = field(
        default=1, metadata={"help": "number of shards to split the dataset into"}
    )

    dataset_shard_index: int = field(
        default=0, metadata={"help": "shard index to use, to be used with dataset_number_of_shards"}
    )

    train_group_size: int = field(
        default=8, metadata={"help": "number of passages used to train for each query"}
    )

    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage for training"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first n negative passages for training"})

    encode_is_query: bool = field(default=False)
    encode_output_path: str = field(default=None, metadata={"help": "where to save the encode"})


    query_max_len: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    passage_max_len: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    query_prefix: str = field(
        default='', metadata={"help": "prefix or instruction for query"}
    )

    passage_prefix: str = field(
        default='', metadata={"help": "prefix or instruction for passage"}
    )

    append_eos_token: bool = field(
        default=False, metadata={"help": "append eos token to query and passage, this is currently used for repllama"}
    )

    pad_to_multiple_of: Optional[int] = field(
        default=16,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is especially useful to "
                    "enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        },
    )

    num_proc: int = field(
        default=1, metadata={"help": "number of processes to use for loading the dataset"}
    )

    encode_text: bool = field(
        default=False, metadata={"help": "whether to encode text or not"}
    )
    encode_image: bool = field(
        default=False, metadata={"help": "whether to encode image or not"}
    )
    encode_audio: bool = field(
        default=False, metadata={"help": "whether to encode audio or not"}
    )
    encode_video: bool = field(
        default=False, metadata={"help": "whether to encode video or not"}
    )
    encode_transcript: bool = field(
        default=False, metadata={"help": "whether to encode transcript or not"}
    )
    encode_modalities: json.loads = field(
        default=None,
        metadata={
            "help": (
                "Per-role modality toggles. Expected structure: "
                "{'default': {'text': bool, ...}, 'query': {...}, 'passage': {...}}. "
                "Values override the legacy encode_* flags when provided."
            )
        },
    )

    def resolve_modalities(self, is_query: bool) -> Dict[str, bool]:
        role = "query" if is_query else "passage"
        resolved = {
            "text": self.encode_text,
            "image": self.encode_image,
            "audio": self.encode_audio,
            "video": self.encode_video,
            "transcript": self.encode_transcript,
        }
        if self.encode_modalities:
            default_config = self.encode_modalities.get("default")
            if default_config:
                for key, value in default_config.items():
                    resolved[key] = bool(value)
            role_config = self.encode_modalities.get(role)
            if role_config:
                for key, value in role_config.items():
                    resolved[key] = bool(value)
        return resolved

    def should_encode_modality(self, modality: str, is_query: bool) -> bool:
        return bool(self.resolve_modalities(is_query).get(modality, False))

@dataclass
class EvaluateArguments:
    output_path: str = field(
        default=None, metadata={"help": "Path to the output file"}
    )

    index_path: str = field(
        default=None, metadata={"help": "Path to the index"}
    )

    is_pickle: bool = field(
        default=False, metadata={"help": "whether to load from pickle or not"}
    )

    index_type: str = field(
        default=None, metadata={"help": "Type of index to evaluate"}
    )
    
    device: str = field(
        default="cuda", metadata={"help": "Device to use for evaluation"}
    )

    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for evaluation"}
    )

    top_k: List[int] = field(
        default_factory=lambda: [1, 5, 10, 100],
        metadata={"help": "Top k for evaluation"}
    )

    output_query_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to output the embedding of queries or not"}
    )

    output_rank_file: bool = field(
        default=False,
        metadata={"help": "Whether to output the rank file or not"}
    )
    

@dataclass
class IndexArguments:
    index_type: str = field(
        default="flat",
        metadata={"help": "Type of index to build"}
    )
    index_output_path: str = field(
        default=None, metadata={"help": "Directory to save the index"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for encoding"}
    )
    device: str = field(
        default="cuda", metadata={"help": "Device to use for encoding"}
    )
    
    # vLLM specific arguments
    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Number of GPUs to use for tensor parallelism with vLLM"}
    )
    gpu_memory_utilization: float = field(
        default=0.9, metadata={"help": "GPU memory utilization for vLLM (0.0 to 1.0)"}
    )
    max_model_len: Optional[int] = field(
        default=None, metadata={"help": "Maximum model length for vLLM"}
    )

    output_pickle: bool = field(
        default=False, metadata={"help": "Whether to output the index as a pickle file or not"}
    )