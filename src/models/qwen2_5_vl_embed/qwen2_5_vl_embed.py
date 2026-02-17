import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
import einops

from transformers.utils import can_return_tuple, auto_docstring, TransformersKwargs
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack

from src.models.qwen2_5_vl_embed.modeling_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLDecoderLayer

class Qwen2_5ForEmbedding(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config, bidirectional=True):
        self.bidirectional = bidirectional
        super().__init__(config)
        self.padding_side = "left"
        # Initialize weights and apply final processing, change the model to bidirectional
        self.post_init()

    def post_init(self):
        super().post_init()
        if self.bidirectional:
            for layer in self.model.language_model.layers:
                layer.self_attn.is_causal = False

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return outputs

        # hidden_states = outputs[0]

        # # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])

        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(
        #         logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
        #     )

        # return Qwen2_5_VLCausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        #     rope_deltas=outputs.rope_deltas,
        # )


class Qwen2_5ForMLMMAE(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Create a dedicated decoder layer for MAE
        self.mae_decoder = Qwen2_5_VLDecoderLayer(config, layer_idx=0)
        
        # Output projection layer
        self.mae_proj = nn.Linear(config.hidden_size, 1176 * 4, bias=True)
        
        self.fused_linear_ce = config.fused_linear_ce if hasattr(config, 'fused_linear_ce') else False
        
        super().post_init()
        
        # Set bidirectional mode (for all layers)
        for layer in self.model.layers:
            layer.self_attn.is_causal = False
        
        # Ensure MAE decoder is also bidirectional
        self.mae_decoder.self_attn.is_causal = False
        
    
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                if (image_nums + video_nums) != 0:
                    remain_images, remain_videos = image_nums, video_nums
                    for _ in range(image_nums + video_nums):
                        if image_token_id in input_tokens and remain_images > 0:
                            ed_image = input_tokens.index(image_token_id, st)
                        else:
                            ed_image = len(input_tokens) + 1
                        if video_token_id in input_tokens and remain_videos > 0:
                            ed_video = input_tokens.index(video_token_id, st)
                        else:
                            ed_video = len(input_tokens) + 1
                        if ed_image < ed_video:
                            t, h, w = (
                                image_grid_thw[image_index][0],
                                image_grid_thw[image_index][1],
                                image_grid_thw[image_index][2],
                            )
                            second_per_grid_t = 0
                            image_index += 1
                            remain_images -= 1
                            ed = ed_image

                        else:
                            t, h, w = (
                                video_grid_thw[video_index][0],
                                video_grid_thw[video_index][1],
                                video_grid_thw[video_index][2],
                            )
                            if second_per_grid_ts is not None:
                                second_per_grid_t = second_per_grid_ts[video_index]
                            else:
                                second_per_grid_t = 1.0
                            video_index += 1
                            remain_videos -= 1
                            ed = ed_video
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )
                        text_len = ed - st

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                        time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                        time_tensor_long = time_tensor.long()
                        t_index = time_tensor_long.flatten()

                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                    if st < len(input_tokens):
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        text_len = len(input_tokens) - st
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                    position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                    mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
                else:
                    temp_position_ids = attention_mask[i].long().cumsum(-1) - 1
                    temp_position_ids.masked_fill_(attention_mask[i] == 0, 1)
                    temp_position_ids = temp_position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
                    temp_mrope_position_deltas = torch.max(temp_position_ids) + 1 - attention_mask[i].shape[0]
                    position_ids[..., i, :] = temp_position_ids
                    mrope_position_deltas.append(temp_mrope_position_deltas)
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        mae_pred_mask: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)


            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    None,
                    None,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        
        # MLM loss calculation (if needed)
        if self.fused_linear_ce:
            hidden_states_for_mlm = hidden_states.view(-1, hidden_states.shape[-1])
            lce = LigerFusedLinearCrossEntropyLoss()
            if labels is not None:
                loss = lce(self.lm_head.weight, hidden_states_for_mlm, labels.view(-1))
            else:
                loss = 0.0
        else:
            logits = self.lm_head(hidden_states).view(-1, self.config.vocab_size)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        
        # Generate position embeddings
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # First process the entire hidden state sequence through MAE decoder
        full_sequence_outputs = self.mae_decoder(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            position_embeddings=position_embeddings,
        )
        
        # Get the processed hidden states
        if isinstance(full_sequence_outputs, tuple):
            decoded_hidden_states = full_sequence_outputs[0]
        else:
            decoded_hidden_states = full_sequence_outputs
        
        # Then extract the token features that need to be predicted
        masked_features = decoded_hidden_states[mae_pred_mask]


        patch_pred = self.mae_proj(masked_features)

        # Reshape to match the target
        patch_pred = einops.rearrange(patch_pred, "b (d p) -> (b p) d", p=4)

        # Calculate MAE loss
        mae_loss = (patch_pred - pixel_labels) ** 2
        mae_loss = mae_loss.mean()
        
        return loss, mae_loss