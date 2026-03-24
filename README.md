# OmniColPress

<p align="center">
  <a href="https://arxiv.org/abs/2602.21202"><img src="https://img.shields.io/badge/arXiv-2602.21202-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/collections/hltcoe/multi-vector-index-compression-in-any-modality"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Collection-yellow" alt="HuggingFace"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

**A modular framework for training and inference of (compressed) multi-vector retrieval across any modality.**

OmniColPress enables late-interaction retrieval across video, image, audio, and text modalities using multimodal foundation models (Qwen2.5-VL, Qwen2.5-Omni, Qwen3-VL, etc.). It implements several representation compression methods to reduce the storage and computational cost of multi-vector retrieval while preserving retrieval quality.

---

## Table of Contents

- [Features](#features)
- [Environment Setup](#environment-setup)
- [Training](#training)
  - [Quick Start](#quick-start)
  - [Training Configuration](#training-configuration)
- [Inference (Evaluation)](#inference-evaluation)
  - [Quick Start](#quick-start-1)
  - [Step 1: Index Building](#step-1-index-building)
  - [Step 2: Retrieval](#step-2-retrieval-evaluation)
  - [Index Types](#index-types)
- [Data Format](#data-format)
- [Notes](#notes)
- [Citation](#citation)

---

## Features

- **Omni-modal retrieval** — Supports text, image, video, and audio for both queries and documents, with per-role modality control.
- **Multiple compression methods** — Attention-Guided Clustering (AGC) with Learned Universal Query tokens, Memory Tokens, Hierarchical Pooling, and Sequence Resizing.
- **Modular architecture** — Every component — dataset, dataloader, collator, processor, model, index, compression method, single/multi-vector mode — is fully modularized and easy to configure or extend.
- **Efficient training** — LoRA, DeepSpeed ZeRO, gradient checkpointing, half-precision training, memory-efficient distributed contrastive loss, and mixed-precision similarity computation — carefully optimized and well supported.
- **Built-in evaluation** — In-training validation with Recall@k, NDCG@k, and MRR metrics, plus standalone index building (MultiVec-Flat, Faiss, Fast-Plaid) and retrieval evaluation.
- **Flexible data recipes** — Multi-dataset training via YAML config with per-dataset validation splits, HuggingFace datasets, and local file formats (JSON, CSV, Parquet).
- **Smooth workflow** — Multi-GPU training with WandB logging, working seamlessly out of the box.

---

## Environment Setup

```bash
conda create -n omnicolpress python=3.11
conda activate omnicolpress

pip install torch torchvision
pip install triton
pip install transformers
pip install deepspeed
pip install ninja
pip install flash-attn --no-build-isolation
pip install peft
pip install librosa
pip install numpy
pip install qwen-vl-utils[decord]
pip install qwen-omni-utils[decord] -U
pip install fast-plaid  # see note below
conda install -c pytorch -c nvidia faiss-gpu
```

> **Note:** Specify the version of `fast-plaid` according to your PyTorch version. See [fast-plaid](https://github.com/lightonai/fast-plaid) for details.

---

## Training

### Quick Start

Training can be launched using either a YAML/JSON configuration file or command-line arguments.

**Multi-GPU with DeepSpeed:**

```bash
deepspeed --num_gpus=4 src/train.py \
  --deepspeed deepspeed/ds_zero3_config.json \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --train_yaml train_config.yaml \
  --encode_modalities ${SEE_EXAMPLE_BELOW} \
  --pooling select \
  --num_appending_token 32 \
  --use_parametric_appending_tokens \
  --output_dir outputs/my_experiment \
  --per_device_train_batch_size 28 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --bf16 \
  --gradient_checkpointing
```

### Training Configuration

The following sections outline the key arguments for training. Arguments marked with `*` need to be specifically modified for each variant of method, while others typically remain unchanged once configured.

#### 1. Model / Processor *

| Argument | Description |
|----------|-------------|
| `model_name_or_path` | Path to pretrained model or HuggingFace model identifier |
| `processor_name_or_path` | Path to pretrained processor (optional, defaults to model path) |

#### 2. Dataset *

**Option A — YAML config (recommended for multi-dataset or reusable configs):**

| Argument | Description |
|----------|-------------|
| `train_yaml` | Path to a YAML file defining one or more training datasets |

Each entry under `datasets:` supports:

| Key | Description |
|-----|-------------|
| `train_path` | Path to training data |
| `corpus_path` | Path to corpus |
| `corpus_assets_path` | Path to related assets (video/image/audio directory) |
| `dataset_name` / `corpus_name` | Data format (`"json"`, `"parquet"`) or a HuggingFace dataset path |
| `is_split` | Set to `true` if providing separate `train_path` and `valid_path` |
| `valid_path` | Path to validation data (when `is_split: true`) |

Example YAML:

```yaml
datasets:
  - train_path: path/to/train.jsonl
    corpus_path: path/to/corpus.jsonl
    corpus_assets_path: path/to/videos/
    dataset_name: json
    corpus_name: json
```

For pre-split datasets (separate train/validation files):

```yaml
datasets:
  - is_split: true
    train_path: path/to/train.jsonl
    valid_path: path/to/valid.jsonl
    corpus_path: path/to/corpus.jsonl
    corpus_assets_path: path/to/videos/
    dataset_name: json
    corpus_name: json
```

Multiple entries are supported, and mixing pre-split and unsplit datasets is allowed.

**Option B — Direct arguments:**

| Argument | Description |
|----------|-------------|
| `dataset_name` | HuggingFace dataset name (default: `"json"`) |
| `dataset_path` | Path to local data files or directory |
| `corpus_name` | HuggingFace dataset name for corpus |
| `corpus_path` | Path to local corpus files or directory |
| `assets_path` | Path to assets (videos, images, audio) for corpus |

#### 3. Data Arguments

| Argument | Description |
|----------|-------------|
| `query_prefix` | Prefix or instruction for query (e.g., `"Query: "`) |
| `passage_prefix` | Prefix or instruction for passage (e.g., `"Passage: "`) |
| `encode_modalities` * | JSON string specifying which modalities to encode for query/passage (see example below) |
| `query_max_len` | Maximum sequence length for query (text only) |
| `passage_max_len` | Maximum sequence length for passage (text only) |
| `train_group_size` | Number of passages per query during training |

**`encode_modalities` example:**

```json
{
  "default": {"text": true, "image": false, "video": true, "audio": false},
  "query": {"video": false},
  "passage": {"text": true, "video": true}
}
```

#### 4. Pooling Methods and Related Arguments *

The framework supports multiple compression methods:

**AGC (Attention-Guided Clustering):**

| Argument | Description |
|----------|-------------|
| `pooling` | Set to `"select"` |
| `num_repr_vectors` | Number of output representation vectors (cluster centroids) |
| `num_appending_token` | Number of Learned Universal Query tokens (e.g., 32) |
| `use_parametric_appending_tokens` | Use learnable parametric tokens (`True`/`False`) |
| `use_cluster_pooling` | Enable Attention-Guided Clustering (`True`/`False`) |
| `use_attn_weight_cluster_pooling` | Use saliency scores to weight AGC pooling (`True`/`False`) |
| `cluster_centroid_weight` | Weight for centroids in non-AGC pooling (default: 1.0; >1.0 = more weight to centroids) |

**Memory Tokens:**

| Argument | Description |
|----------|-------------|
| `pooling` | Set to `"memory"` |
| `num_appending_token` | Number of memory tokens |
| `use_parametric_appending_tokens` | Use learnable parametric tokens |

**Hierarchical Pooling:**

| Argument | Description |
|----------|-------------|
| `pooling` | Set to `"hierarchical_clustering"` |
| `num_repr_vectors` | Number of representation vectors to retain |
| `num_protected_tokens` | Number of protected tokens (default: 1) |
| `protected_tokens_position` | Which end to protect tokens from: `"first"` protects the first N tokens, `"last"` protects the last N tokens (default: `"first"`) |

**Sequence Resizing:**

| Argument | Description |
|----------|-------------|
| `pooling` | Set to `"resize"` |
| `resizer_input_size` | Fixed input sequence length |
| `resizer_output_size` | Output vector count (sequence dimension) |
| `resizer_hidden_size` | Hidden size of MLP (optional; uses single linear if `None`) |

**Full ColBERT (no compression):**

| Argument | Description |
|----------|-------------|
| `pooling` | Set to `"colbert"` |

**Other:**

| Argument | Description |
|----------|-------------|
| `normalize` | Normalize query and passage representations (default: `True`) |

#### 5. Evaluation (Validation) Arguments

| Argument | Description |
|----------|-------------|
| `index_type` | Index type for evaluation (`"flat"`, `"multivec"`, `"fast-plaid"`) |
| `do_eval` | Enable evaluation during training (`True`/`False`) |
| `validation_split_ratio` | Fraction of training data to reserve for validation (0–1) |
| `eval_steps` | Number of training steps between evaluations |
| `eval_strategy` | Evaluation strategy (`"steps"`, `"epoch"`, `"no"`) |

#### 6. GPU Memory / Performance *

| Argument | Description |
|----------|-------------|
| `per_device_train_batch_size` | Training batch size per device |
| `gradient_accumulation_steps` | Number of gradient accumulation steps |
| `per_device_eval_batch_size` | Evaluation batch size per device |

#### 7. Training Hyperparameters *

| Argument | Description |
|----------|-------------|
| `learning_rate` | Learning rate |
| `weight_decay` | Weight decay for regularization |
| `num_train_epochs` | Number of training epochs |
| `warmup_ratio` | Warmup ratio |
| `seed` | Random seed for reproducibility |
| `temperature` | Temperature for softmax of the logits (default: `1.0`) |

#### 8. Additional Training Options

| Argument | Description |
|----------|-------------|
| `bf16` | Use bfloat16 precision (or `fp16` for float16) |
| `gradient_checkpointing` | Enable gradient checkpointing to save memory |
| `attn_implementation` | Attention implementation (`"flash_attention_2"`, `"sdpa"`, `"eager"`) |
| `lora` | Enable LoRA parameter-efficient fine-tuning |
| `lora_r` | LoRA rank (default: 16) |
| `lora_alpha` | LoRA alpha (default: 64) |
| `lora_dropout` | LoRA dropout (default: 0.1) |
| `lora_target_modules` | Comma-separated list of target modules for LoRA |

#### 9. Logging

| Argument | Description |
|----------|-------------|
| `report_to` | Logging backend (e.g., `"wandb"`) |
| `logging_first_step` | Log the first training step |
| `logging_steps` | Number of steps between logging |
| `run_name` | Name for the run (used in W&B) |

#### 10. DeepSpeed Configuration

- DeepSpeed ZeRO Stage 3 configuration is available at `deepspeed/ds_zero3_config.json`.
- Use `--deepspeed` flag to enable DeepSpeed training.

---

## Inference (Evaluation)

Evaluation consists of two separate steps: **Index Building** (encoding corpus documents) and **Retrieval** (encoding queries and searching the index).

### Quick Start

```bash
# Step 1: Build Index (encode corpus)
torchrun --nproc_per_node=4 -m src.build_index \
    --model_name_or_path path/to/checkpoint \
    --corpus_path path/to/corpus.jsonl \
    --assets_path path/to/videos/ \
    --index_output_path path/to/index \
    --index_type multivec \
    --encode_modalities '{"default":{"text":true,"video":true}}' \
    --batch_size 4
    # ... pooling args (must match training)

# Step 2: Retrieve & Evaluate (encode queries, search index)
torchrun --nproc_per_node=4 -m src.evaluate \
    --model_name_or_path path/to/checkpoint \
    --query_path path/to/queries.csv \
    --qrels_path path/to/judgments.jsonl \
    --index_path path/to/index \
    --index_type multivec \
    --encode_is_query \
    --encode_modalities '{"default":{"text":true}}' \
    --batch_size 8 \
    --output_path path/to/results \
    --top_k 1 5 10 100
    # ... pooling args (must match training)
```

> **Important:** Model and pooling arguments must be identical to the training configuration. See [Pooling Methods](#4-pooling-methods-and-related-arguments-) for details.

### Step 1: Index Building

Encodes all corpus documents and builds a searchable index.

```bash
torchrun --nproc_per_node=NUM_GPUS -m src.build_index [arguments]
```

#### Index Building Arguments

**Shared arguments:** Model, pooling, and data prefix/length arguments are the same as training. Ensure they match your trained model's configuration.

| Argument | Description |
|----------|-------------|
| `lora_name_or_path` | Path to a pretrained LoRA adapter (required when the model was trained with LoRA) |

**Data arguments:**

| Argument | Description |
|----------|-------------|
| `corpus_path` * | Path to corpus file |
| `assets_path` * | Path to media assets (videos/images/audio) |
| `dataset_name` | Dataset format (`"json"`, `"csv"`, `"parquet"`) |
| `encode_modalities` | JSON string for fine-grained modality control (see [Data Arguments](#3-data-arguments)) |

**Index arguments:**

| Argument | Description |
|----------|-------------|
| `index_output_path` * | Directory to save the index |
| `index_type` * | Index type (`"flat"`, `"multivec"`, `"fast-plaid"`) |
| `batch_size` | Encoding batch size |

### Step 2: Retrieval (Evaluation)

Encodes queries, searches against the index, and computes metrics.

```bash
torchrun --nproc_per_node=NUM_GPUS -m src.evaluate [arguments]
```

#### Retrieval Arguments

**Shared arguments:** Same as index building (must use identical model/pooling configuration).

**Data arguments:**

| Argument | Description |
|----------|-------------|
| `query_path` * | Path to queries file |
| `qrels_path` * | Path to relevance judgments (ground truth) |
| `encode_is_query` | Flag indicating query encoding mode |
| `encode_modalities` | JSON string for fine-grained modality control (see [Data Arguments](#3-data-arguments)) |

**Evaluation arguments:**

| Argument | Description |
|----------|-------------|
| `index_path` * | Path to the built index |
| `index_type` * | Index type (must match index building) |
| `output_path` | Directory to save evaluation results |
| `batch_size` | Query encoding batch size |
| `top_k` | List of k values for Recall@k and NDCG@k (e.g., `1 5 10 100`) |

### Index Types

| Index Type | Description | Use Case |
|------------|-------------|----------|
| `flat` | Single-vector FAISS index | Single-vector pooling methods |
| `multivec` | Multi-vector index (PyTorch-based) | Multi-vector retrieval (ColBERT, AGC, Memory Tokens, Sequence Resizing) |
| `fast-plaid` | Fast-Plaid multi-vector index | Multi-vector retrieval ~~*(small corpora only; see [fast-plaid issue #27](https://github.com/lightonai/fast-plaid/issues/27))*~~ <br>**Update:** Thanks to the fantastic work by the LightOn team in [v1.4.5](https://github.com/lightonai/fast-plaid/releases/tag/1.4.5) ([PR #39](https://github.com/lightonai/fast-plaid/pull/39)), the PyTorch sort-based `quantile()` bottleneck was elegantly resolved by switching to a `kthvalue` approach. Fast-Plaid can now scale to corpora of essentially any size (including [Multivent2.0](https://huggingface.co/datasets/hltcoe/MultiVENT2.0)), and is highly recommended! |

---

## Data Format

### Training Data

Each line in the training file is a JSON object:

```json
{
  "query_id": "q1",
  "query": "What is this video about?",
  "query_image": "path/to/image.jpg",
  "query_video": "path/to/video.mp4",
  "query_audio": "path/to/audio.wav",
  "positive_document_ids": ["doc1", "doc2"],
  "negative_document_ids": ["doc3", "doc4"]
}
```

Media fields (`query_image`, `query_video`, `query_audio`) are optional — include only the modalities relevant to your task.

### Corpus Data

Each line in the corpus file is a JSON object:

```json
{
  "docid": "doc1",
  "text": "Document text content",
  "title": "Document title",
  "image": "relative/path/to/image.jpg",
  "video": "relative/path/to/video.mp4",
  "audio": "relative/path/to/audio.wav"
}
```

Media paths are relative to `assets_path` (or `corpus_assets_path` in YAML config). Fields `title`, `image`, `video`, and `audio` are optional.

---

## Notes

- Arguments marked with `*` generally need to be modified for each experiment.
- Model and pooling arguments must be identical across training, index building, and evaluation.
- The framework supports both single-dataset and multi-dataset training (via `train_yaml`).
---

## Citation

If you find OmniColPress useful, please cite:

```bibtex
@misc{qin2026multivectorindexcompressionmodality,
      title={Multi-Vector Index Compression in Any Modality}, 
      author={Hanxiang Qin and Alexander Martin and Rohan Jha and Chunsheng Zuo and Reno Kriz and Benjamin Van Durme},
      year={2026},
      eprint={2602.21202},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2602.21202}, 
}
```
