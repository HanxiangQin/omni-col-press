import os
from dataclasses import dataclass
from typing import Dict, Optional, List

from scipy.cluster import hierarchy

import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers import PreTrainedModel, AutoModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file

from transformers.file_utils import ModelOutput
from src.arguments import ModelArguments, TrainingArguments
from src.utils import dist_gather_tensor, dist_gather_tensor_with_grads

from accelerate.utils import pad_across_processes

import logging
logger = logging.getLogger(__name__)

@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    q_mask: Optional[Tensor] = None
    p_mask: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderModel(nn.Module):
    def __init__(self,
                 model: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 model_args: ModelArguments = None,
                 ):
        super().__init__()
        self.config = model.config
        self.model = model
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.model_args = model_args
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.return_loss = False

    def forward(
        self,
        query_inputs: Dict[str, Tensor] = None,
        passage_inputs: Dict[str, Tensor] = None,
        passage_ids: Optional[Tensor] = None,
    ):
        q_reps, q_mask = self.encode_query(query_inputs) if query_inputs else None
        p_reps, p_mask = self.encode_passage(passage_inputs) if passage_inputs else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps,
                q_mask=q_mask,
                p_mask=p_mask
            )

        should_compute_loss = self.training or self.return_loss
        if should_compute_loss:
            group_size = p_reps.size(0) // q_reps.size(0)
            group_size = max(group_size, 1)
            all_passage_ids = passage_ids
            all_p_reps = p_reps
            all_p_masks = p_mask
            if self.is_ddp:
                # q_reps = dist_gather_tensor(q_reps, self.world_size, self.process_rank)
                all_p_reps = dist_gather_tensor_with_grads(p_reps, self.world_size, self.process_rank)
                # q_mask = dist_gather_tensor(q_mask, self.world_size, self.process_rank)
                if p_mask is not None:
                    all_p_masks = dist_gather_tensor(p_mask, self.world_size, self.process_rank)
                if passage_ids is not None:
                    all_passage_ids = dist_gather_tensor(passage_ids, self.world_size, self.process_rank)

            scores = self.compute_similarity(q_reps, all_p_reps, q_mask, all_p_masks)
            scores = scores.view(q_reps.size(0), -1)

            if passage_ids is not None:
                target = self._build_multi_positive_target(
                    group_size=group_size,
                    local_passage_ids=passage_ids,
                    all_passage_ids=all_passage_ids,
                    device=scores.device,
                    dtype=scores.dtype
                )
            else:
                target = self._build_default_target(
                    batch_size=scores.size(0),
                    group_size=group_size,
                    device=scores.device,
                    process_rank=getattr(self, "process_rank", 0),
                )

            logits = scores / self.temperature
            loss = self.compute_loss(logits, target)
        else:
            scores = self.compute_similarity(q_reps, p_reps, q_mask, p_mask)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
            q_mask=q_mask,
            p_mask=p_mask,
        )

    def _trim_embeddings_by_mask(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        if attention_mask is None:
            return embeddings.unbind(dim=0)
        filtered_list = []
        for i in range(embeddings.shape[0]):
            mask = attention_mask[i].bool()
            filtered_list.append(embeddings[i][mask])
        return filtered_list

    def _pooling_hierarchical_clustering(
        self,
        embeddings_list: list[torch.Tensor],
        pool_factor: int = None,
        num_clusters: int = None,
        protected_tokens: int = 1,
        protected_tokens_position: str = "first",
    ) -> list[torch.Tensor]:
        device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
        pooled_embeddings_list = []

        for embedding in embeddings_list:
            embedding = embedding.to(device=device)

            # Separate protected tokens from the rest
            if protected_tokens > 0 and protected_tokens_position == "first":
                protected_embeddings = embedding[:protected_tokens]
                embeddings_to_pool = embedding[protected_tokens:]
            elif protected_tokens > 0:
                protected_embeddings = embedding[-protected_tokens:]
                embeddings_to_pool = embedding[:-protected_tokens]
            else:
                protected_embeddings = []
                embeddings_to_pool = embedding

            # Compute cosine similarity and convert to distance matrix
            cosine_similarities = torch.mm(
                input=embeddings_to_pool, mat2=embeddings_to_pool.t()
            )
            distance_matrix = 1 - cosine_similarities.float().cpu().numpy()

            # Perform hierarchical clustering using Ward's method
            clusters = hierarchy.linkage(distance_matrix, method="ward")
            num_embeddings = len(embeddings_to_pool)

            # Determine the number of clusters
            if pool_factor is not None:
                num_clusters = num_embeddings // pool_factor
            elif num_clusters is not None:
                num_clusters = num_clusters
            num_clusters = max(num_clusters, 1)
            cluster_labels = hierarchy.fcluster(
                clusters, t=num_clusters, criterion="maxclust"
            )

            # Pool embeddings within each cluster
            pooled_embeddings = []
            for cluster_id in range(1, num_clusters + 1):
                cluster_indices = torch.where(
                    condition=torch.tensor(
                        data=cluster_labels == cluster_id, device=device
                    )
                )[0]
                if cluster_indices.numel() > 0:
                    cluster_embedding = embeddings_to_pool[cluster_indices].mean(dim=0)
                    pooled_embeddings.append(cluster_embedding)

            # Re-insert protected embeddings at their original position
            if protected_tokens_position == "first":
                pooled_embeddings = list(protected_embeddings) + pooled_embeddings
            else:
                pooled_embeddings.extend(protected_embeddings)
            pooled_embeddings_list.append(torch.stack(tensors=pooled_embeddings))

        return pooled_embeddings_list
        
    def _pooling(self,
        hidden_states: List[torch.Tensor],
        attention_mask: torch.Tensor,
        is_query: bool = False
        ):
        last_hidden_state = hidden_states[-1]
        if self.pooling in ['hierarchical_clustering']:
            if is_query:
                reps = last_hidden_state
            else:
                trimmed_embeddings = self._trim_embeddings_by_mask(last_hidden_state, attention_mask)
                reps_list = self._pooling_hierarchical_clustering(
                    embeddings_list=trimmed_embeddings,
                    num_clusters=self.model_args.get_num_clusters(),
                    protected_tokens=self.model_args.get_num_protected_tokens(),
                    protected_tokens_position=self.model_args.protected_tokens_position,
                )
                reps = torch.stack(reps_list, dim=0)
                attention_mask = None
        elif self.pooling in ['memory']:
            if is_query:
                reps = last_hidden_state
            else:
                left_padding = attention_mask is not None and (attention_mask[:, -1].sum() == attention_mask.shape[0])
                if left_padding or attention_mask is None:
                    reps = last_hidden_state[:, -self.model_args.num_appending_token:, :]
                else:
                    num_appending_token = min(self.model_args.num_appending_token, last_hidden_state.shape[1])
                    sequence_lengths = attention_mask.sum(dim=1).to(torch.long)
                    start_positions = (sequence_lengths - num_appending_token).clamp(min=0)
                    offsets = torch.arange(num_appending_token, device=last_hidden_state.device)
                    positions = (start_positions[:, None] + offsets).clamp(max=last_hidden_state.shape[1] - 1)
                    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)[:, None]
                    reps = last_hidden_state[batch_indices, positions]
                attention_mask = None
        elif self.pooling in ['colbert']:
            reps = last_hidden_state
        elif self.pooling in ['cls', 'first']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                seq_len = attention_mask.shape[1]
                positions = seq_len - attention_mask.sum(dim=1).to(torch.long)
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), positions]
            else:
                reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                reps = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps, attention_mask

    def encode_passage(self, psg):
        raise NotImplementedError("encode_passage is not implemented as EncoderModel is a abstract class")

    def encode_query(self, qry):
        raise NotImplementedError("encode_query is not implemented as EncoderModel is a abstract class")

    def compute_similarity(self, q_reps, p_reps, q_mask=None, p_mask=None):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    @staticmethod
    def _mask_to_soft_targets(positive_mask: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
        positive_targets = positive_mask.to(device=device, dtype=dtype)
        normalization = positive_targets.sum(dim=1, keepdim=True)
        positive_targets = positive_targets / normalization
        return positive_targets

    def _build_multi_positive_target(self, group_size: int, local_passage_ids: Tensor, all_passage_ids: Tensor, device: torch.device, dtype: torch.dtype):
        flattened_all_passage_ids = all_passage_ids.view(-1).contiguous()
        local_passage_ids = local_passage_ids.contiguous().view(-1, group_size)
        local_positive_passage_ids = local_passage_ids[:, 0]

        positive_mask = local_positive_passage_ids.unsqueeze(1).eq(flattened_all_passage_ids.unsqueeze(0))
        return self._mask_to_soft_targets(positive_mask, device=device, dtype=dtype)

    def _build_default_target(
        self,
        batch_size: int,
        group_size: int,
        device: torch.device,
        process_rank: int = 0,
    ):
        positive_indices = torch.arange(batch_size, device=device, dtype=torch.long)
        positive_indices = positive_indices * group_size
        rank_offset = process_rank * batch_size * group_size
        if rank_offset:
            positive_indices = positive_indices + rank_offset
        return positive_indices

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    @classmethod
    def _get_embedding_layer(cls, base_model: PreTrainedModel) -> nn.Module:
        embed_layer = embed_name = None
        if hasattr(base_model, 'get_input_embeddings'):
            embed_layer = base_model.get_input_embeddings()
        if embed_layer is None:
            raise ValueError("Could not find embedding layer")
        
        for name, module in base_model.named_modules():
            if module is embed_layer:
                embed_name = name
        if embed_name is None:
            raise ValueError("Could not find name of embedding layer")
        return embed_layer, embed_name
        

    @classmethod
    def build(
            cls,
            transformer_cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            tokenizer_len: int = 0,
            **hf_kwargs,
    ):  
        base_model = transformer_cls.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        
        # Resize embeddings if tokenizer_len is provided and larger than current vocab size
        embed_layer, embed_name = cls._get_embedding_layer(base_model)
        current_vocab_size = embed_layer.weight.shape[0]
        needs_resize = tokenizer_len > 0 and tokenizer_len > current_vocab_size
        if needs_resize:
            base_model.resize_token_embeddings(tokenizer_len)

        if train_args.gradient_checkpointing:
            base_model.gradient_checkpointing_enable(train_args.gradient_checkpointing_kwargs)
        
        if model_args.lora or model_args.lora_name_or_path:
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                # Add modules_to_save if tokenizer was resized (appending tokens are used)
                modules_to_save = [embed_name] if needs_resize else None
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    modules_to_save=modules_to_save,
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            trainable_params, all_param = lora_model.get_nb_trainable_parameters()
            instance = cls(
                model=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                model_args=model_args,
            )
        else:
            instance = cls(
                model=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                model_args=model_args,
            )
        return instance

    @classmethod
    def load(
        cls,
        transformer_cls,
        model_args: ModelArguments,
        tokenizer_len: int = 0,
        **hf_kwargs,
    ):
        pooling = model_args.pooling
        normalize = model_args.normalize
        lora_name_or_path = model_args.lora_name_or_path
        base_model = transformer_cls.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        
        # Resize embeddings if tokenizer_len is provided and larger than current vocab size
        embed_layer, embed_name = cls._get_embedding_layer(base_model)
        current_vocab_size = embed_layer.weight.shape[0]
        if tokenizer_len > 0 and tokenizer_len != current_vocab_size:
            base_model.resize_token_embeddings(tokenizer_len)
        
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            instance = cls(
                model=lora_model,
                pooling=pooling,
                normalize=normalize,
                model_args=model_args,
            )
        else:
            instance = cls(
                model=base_model,
                pooling=pooling,
                normalize=normalize,
                model_args=model_args,
            )
        # Load encoder-specific modules if they were saved separately
        extra_state_path = None
        candidate_paths = []
        # 1) explicit override
        if getattr(model_args, "extra_state_path", None):
            candidate_paths.append(model_args.extra_state_path)
        # 2) alongside base model path
        candidate_paths.append(os.path.join(model_args.model_name_or_path, "extra_encoder_state.safetensors"))
        # 3) alongside LoRA adapter path (common when using PEFT checkpoints)
        if model_args.lora_name_or_path:
            candidate_paths.append(os.path.join(model_args.lora_name_or_path, "extra_encoder_state.safetensors"))

        for cand in candidate_paths:
            if cand and os.path.isfile(cand):
                extra_state_path = cand
                break

        if extra_state_path:
            extra_state = safe_load_file(extra_state_path, device="cpu")
            # Merge the saved extra state on top of the freshly loaded model weights.
            # This prevents load_state_dict from complaining about the main model weights
            # (they are already loaded from HF), while still restoring auxiliary modules
            # such as resizers or projection heads.
            merged_state = instance.state_dict()
            merged_state.update(extra_state)
            missing, unexpected = instance.load_state_dict(merged_state, strict=False)
            if missing:
                logger.warning("Missing keys while loading extra encoder state: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys while loading extra encoder state: %s", unexpected)

        return instance

    def save(self, output_dir: str, state_dict_override: dict | None = None):
        os.makedirs(output_dir, exist_ok=True)
        state_dict = state_dict_override if state_dict_override is not None else self.state_dict()
        model_prefix = "model."
        model_state = {}
        extra_state = {}
        for k, v in state_dict.items():
            if k.startswith(model_prefix):
                model_state[k[len(model_prefix):]] = v
            else:
                extra_state[k] = v

        self.model.save_pretrained(output_dir, state_dict=model_state, safe_serialization=True)
        if extra_state:
            extra_path = os.path.join(output_dir, "extra_encoder_state.safetensors")
            safe_save_file(extra_state, extra_path)