"""Qwen head-camera and language features for the VLA's six coupled action blocks."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig
from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from .qwen35_mot import apply_qwen35_mot_patch, qwen35_layer_type


class QwenBackbone(nn.Module):
    """Head-camera and language backbone for the complete VLA."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        apply_qwen35_mot_patch()
        if config.backbone_config is not None:
            vlm_config = Qwen3_5Config(**config.backbone_config)
        elif config.tokenizer_path:
            vlm_config = AutoConfig.from_pretrained(config.tokenizer_path, trust_remote_code=False)
        else:
            raise ValueError("Provide tokenizer_path/vlm_path or an embedded backbone_config.")
        if vlm_config.model_type != "qwen3_5":
            raise ValueError("This policy requires a Qwen3.5 backbone.")
        vlm_config._attn_implementation = config.backbone_attention_implementation
        vlm_config.text_config._attn_implementation = config.backbone_attention_implementation
        vlm_config.vision_config._attn_implementation = config.vit_attn_implementation
        self.vlm_config = vlm_config
        config.backbone_config = vlm_config.to_dict()
        self.qwen35 = Qwen3_5ForConditionalGeneration._from_config(vlm_config)
        text = vlm_config.text_config
        action = config.action_config
        if sum(kind == "full_attention" for kind in text.layer_types) != action["num_hidden_layers"] // 3:
            raise ValueError("VLA requires one Qwen full-attention layer per three action layers.")
        if (text.num_key_value_heads, text.head_dim) != (action["num_key_value_heads"], action["head_dim"]):
            raise ValueError("Qwen and action KV heads/head dimensions must match.")
        rope_params = getattr(text, "rope_parameters", None) or {}
        qwen_rope_theta = rope_params.get("rope_theta", getattr(text, "rope_theta", None))
        if qwen_rope_theta is not None and float(qwen_rope_theta) != float(action["rope_theta"]):
            raise ValueError(
                f"Qwen rope_theta ({qwen_rope_theta}) must match action rope_theta ({action['rope_theta']})."
            )
        qwen_rotary_factor = rope_params.get("partial_rotary_factor", getattr(text, "partial_rotary_factor", None))
        if qwen_rotary_factor is not None:
            qwen_rotary_dim = int(round(text.head_dim * float(qwen_rotary_factor)))
            if qwen_rotary_dim != int(action["rotary_dim"]):
                raise ValueError(
                    f"Qwen rotary_dim ({qwen_rotary_dim}) must match action rotary_dim ({action['rotary_dim']})."
                )

    @property
    def language_model(self):
        return self.qwen35.model.language_model

    def embed_images(self, pixel_values: Tensor, image_grid_thw: Tensor) -> Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(f"Qwen image input must be preprocessed [B,N,256,1536], got {pixel_values.shape}.")
        pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        visual = self.qwen35.model.visual
        num_images = image_grid_thw.shape[0]
        visual_input_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else visual.dtype
        output = visual(pixel_values.to(dtype=visual_input_dtype), grid_thw=image_grid_thw, return_dict=True)
        image_embeds = output.pooler_output
        if num_images <= 0 or image_embeds.shape[0] != num_images * 64:
            raise ValueError(
                "Every 256x256 camera image must produce exactly 64 Qwen3.5 "
                f"visual tokens; got {image_embeds.shape[0]} output tokens for "
                f"{num_images} images."
            )
        # Every Qwen head frame has a fixed 256x256 grid.
        # Reshape directly instead of copying the CUDA grid to Python with
        # ``tolist()`` and splitting hundreds of same-sized chunks each step.
        return image_embeds.reshape(num_images, 64, -1)

    def embed_prefix(self, head_tokens, lang_tokens, lang_masks, subtask_tokens=None, subtask_masks=None):
        """Pack head frames, valid task tokens, and optional shifted subtask tokens with right-tail padding only.

        Keeping all padding at the right tail avoids mid-sequence pad slots
        between task and subtask tokens, which would otherwise cause Qwen3.5's
        GatedDeltaNet layers to exponentially decay recurrent state across pad
        tokens via non-zero ``dt_bias``.
        """
        batch, count, patches, _ = head_tokens.shape
        if patches != 64 or count != self.config.history_memory + 1:
            raise ValueError("Expected history_memory + 1 head frames with 64 patches each.")
        image_slots = 66 * count
        if lang_tokens.shape != (batch, 72) or lang_masks.shape != (batch, 72):
            raise ValueError("The VLM task prompt must occupy exactly 72 slots.")
        lang_valid = lang_masks.bool()
        if not torch.compiler.is_compiling() and (lang_valid[:, 1:] & ~lang_valid[:, :-1]).any():
            raise ValueError("lang_masks must be right-padded with contiguous valid tokens from index 0.")
        lang_lens = lang_valid.long().sum(dim=1)
        embedding = self.language_model.embed_tokens
        device = head_tokens.device
        start_ids = torch.full((batch, 1), self.vlm_config.vision_start_token_id, device=device, dtype=torch.long)
        end_ids = torch.full((batch, 1), self.vlm_config.vision_end_token_id, device=device, dtype=torch.long)
        start, end = embedding(start_ids), embedding(end_ids)
        image_hidden = torch.cat(
            [torch.cat([start, head_tokens[:, index], end], dim=1) for index in range(count)],
            dim=1,
        )
        image_mask = torch.ones(batch, image_slots, dtype=torch.bool, device=device)
        prefix_length = image_slots + 72
        prefix_mask = torch.cat([image_mask, lang_valid], dim=1)
        position_start = 10 * count + lang_lens
        lang_emb = embedding(lang_tokens)
        subtask_gather_pos = None

        if subtask_tokens is None:
            hidden = torch.cat([image_hidden, lang_emb], dim=1)
            padding = prefix_mask
        else:
            if subtask_masks is None or subtask_tokens.shape != (batch, 100) or subtask_masks.shape != (batch, 100):
                raise ValueError("Subtask targets and masks must occupy exactly 100 slots.")
            subtask_valid = subtask_masks.bool()
            if not torch.compiler.is_compiling() and (subtask_valid[:, 1:] & ~subtask_valid[:, :-1]).any():
                raise ValueError("subtask_masks must be right-padded with contiguous valid tokens from index 0.")
            # A regular language EOS/BOS starts the prediction region. This is
            # teacher forcing: target i is predicted from target <i, never itself.
            bos = self.vlm_config.text_config.bos_token_id
            if bos is None:
                bos = self.vlm_config.text_config.eos_token_id
            if isinstance(bos, (list, tuple)):
                bos = bos[0]
            if bos is None:
                raise ValueError("Qwen must define a BOS or EOS token for shifted language supervision.")
            shifted = torch.cat([torch.full_like(subtask_tokens[:, :1], bos), subtask_tokens[:, :-1]], dim=1)
            subtask_emb = embedding(shifted)
            hidden_dim = lang_emb.shape[-1]
            tail_hidden = lang_emb.new_zeros((batch, 172, hidden_dim))
            tail_hidden[:, :72] = lang_emb * lang_valid.unsqueeze(-1).to(lang_emb.dtype)
            tail_padding = torch.zeros((batch, 172), dtype=torch.bool, device=device)
            tail_padding[:, :72] = lang_valid
            subtask_offsets = lang_lens[:, None] + torch.arange(100, device=device)[None]
            masked_subtask_emb = subtask_emb * subtask_valid.unsqueeze(-1).to(subtask_emb.dtype)
            tail_hidden = tail_hidden.scatter(
                1,
                subtask_offsets.unsqueeze(-1).expand(-1, -1, hidden_dim),
                masked_subtask_emb,
            )
            tail_padding = tail_padding.scatter(1, subtask_offsets, subtask_valid)
            hidden = torch.cat([image_hidden, tail_hidden], dim=1)
            padding = torch.cat([image_mask, tail_padding], dim=1)
            subtask_gather_pos = image_slots + subtask_offsets

        positions = torch.zeros(3, batch, hidden.shape[1], device=device, dtype=torch.long)
        coordinates = torch.arange(8, device=device)
        for index in range(count):
            slot, offset = index * 66, index * 10
            positions[:, :, slot] = offset
            positions[0, :, slot + 1 : slot + 65] = offset + 1
            positions[1, :, slot + 1 : slot + 65] = coordinates[:, None].expand(-1, 8).reshape(-1) + offset + 1
            positions[2, :, slot + 1 : slot + 65] = coordinates[None, :].expand(8, -1).reshape(-1) + offset + 1
            positions[:, :, slot + 65] = offset + 9
        tail_positions = padding[:, image_slots:].long().cumsum(dim=-1) + (10 * count - 1)
        tail_positions = tail_positions.masked_fill(~padding[:, image_slots:], 0)
        positions[:, :, image_slots:] = tail_positions[None]
        return hidden, padding, positions, prefix_length, prefix_mask, position_start, subtask_gather_pos

    def encode_prefix(self, hidden, padding, positions, prefix_length, *, need_outputs: bool = True):
        """One differentiable VLM pass; retain only image/task KV for MoT."""
        embeddings = self.language_model.rotary_emb(hidden, positions)
        length = hidden.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        allowed = causal[None, None] & padding[:, None, None, :]
        layers = self.language_model.layers
        last_full_idx = max(
            idx for idx, layer in enumerate(layers) if qwen35_layer_type(layer) == "full_attention"
        )
        prefix_kv = []
        for idx, layer in enumerate(layers):
            if qwen35_layer_type(layer) == "linear_attention":
                if not need_outputs and idx > last_full_idx:
                    break
                hidden = layer(hidden, mot_mode="linear", attention_mask=padding)
            else:
                if not need_outputs and idx == last_full_idx:
                    key, value = layer(hidden, mot_mode="project_kv")
                    _, key = apply_rotary_pos_emb(key, key, *embeddings)
                    prefix_kv.append((key[:, :, :prefix_length], value[:, :, :prefix_length]))
                    return None, prefix_kv
                query, key, value, gate = layer(hidden, mot_mode="project")
                query, key = apply_rotary_pos_emb(query, key, *embeddings)
                # Slice before exposing the cache; prefix_mask masks out any
                # compacted subtask slots for samples with short prompts.
                prefix_kv.append((key[:, :, :prefix_length], value[:, :, :prefix_length]))
                attended = (
                    F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=allowed,
                        dropout_p=0.0,
                        scale=layer.self_attn.scaling,
                        enable_gqa=True,
                    )
                    .transpose(1, 2)
                    .reshape(hidden.shape[0], length, -1)
                )
                hidden = layer(hidden, attended, gate, mot_mode="finish")
        return (self.language_model.norm(hidden) if need_outputs else None), prefix_kv
