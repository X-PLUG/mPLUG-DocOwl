import math
from typing import Any, Optional, Tuple, Union

from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from icecream import ic
import einops
from einops import rearrange

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class MplugOwlVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.hidden_size))

        self.pre_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.size(0)
        image_embeds = self.patch_embed(pixel_values)
        image_embeds = image_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.cls_token.expand(batch_size, 1, -1).to(image_embeds.dtype)
        embeddings = torch.cat([class_embeds, image_embeds], dim=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1)].to(image_embeds.dtype)
        embeddings = self.pre_layernorm(embeddings)
        return embeddings



class MplugOwlVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.attention_dropout)

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, seq_len, embed_dim = hidden_states.size()

        mixed_qkv = self.query_key_value(hidden_states)

        mixed_qkv = mixed_qkv.reshape(bsz, seq_len, self.num_heads, 3, embed_dim // self.num_heads).permute(
            3, 0, 2, 1, 4
        )  # [3, b, np, sq, hn]
        query_states, key_states, value_states = (
            mixed_qkv[0],
            mixed_qkv[1],
            mixed_qkv[2],
        )
        # if self.config.use_flash_attn and flash_attn_func is not None:
        if False:
            # [b*sq, np, hn]
            query_states = query_states.permute(0, 2, 1, 3).contiguous()
            query_states = query_states.view(query_states.size(0) * query_states.size(1), query_states.size(2), -1)

            key_states = key_states.permute(0, 2, 1, 3).contiguous()
            key_states = key_states.view(key_states.size(0) * key_states.size(1), key_states.size(2), -1)

            value_states = value_states.permute(0, 2, 1, 3).contiguous()
            value_states = value_states.view(value_states.size(0) * value_states.size(1), value_states.size(2), -1)

            cu_seqlens = torch.arange(
                0, (bsz + 1) * seq_len, step=seq_len, dtype=torch.int32, device=query_states.device
            )

            context_layer = flash_attn_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens,
                cu_seqlens,
                seq_len,
                seq_len,
                self.dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False,
                return_attn_probs=False,
            )
            # [b*sq, np, hn] => [b, sq, np, hn]
            context_layer = context_layer.view(bsz, seq_len, context_layer.size(1), context_layer.size(2))
        else:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

            attention_scores = attention_scores * self.scale

            # Normalize the attention scores to probabilities.
            attention_probs = torch.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.dense(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MplugOwlMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MplugOwlVisionEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MplugOwlVisionAttention(config)
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MplugOwlMLP(config)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    
    
class MplugOwlVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MplugOwlVisionEncoderLayer`].

    Args:
        config (`MplugOwlVisionConfig`):
            The corresponding vision configuration for the `MplugOwlEncoder`.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([MplugOwlVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MplugOwlVisionModel(PreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.embeddings = MplugOwlVisionEmbeddings(config)
        self.encoder = MplugOwlVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

        self.post_init()


    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings


class MplugDocOwlHReducerModel(PreTrainedModel):
    def __init__(self, config, language_hidden_size):
        super().__init__(config)
        self.config = config
        self.ln_q = torch.nn.LayerNorm(self.config.hidden_size, eps=1e-6)
        self.conv_shape = (int(self.config.conv_shape.split('x')[0]), int(self.config.conv_shape.split('x')[1])) # 
        self.conv_patch=self.conv_shape[0]*self.conv_shape[1]
        ## feature interaction with a conv layer
        self.reducer_before = torch.nn.Sequential(
            nn.Conv2d(self.config.hidden_size, self.conv_patch*self.config.hidden_size, kernel_size=self.conv_shape, stride=self.conv_shape, bias=True),
            nn.GELU()
        )
        ## reduce visual feature length with a conv layer
        self.reducer = nn.Conv2d(self.config.hidden_size, self.config.hidden_size, kernel_size=self.conv_shape, stride=self.conv_shape, bias=True)    
        ## align visual features with language embedding with fc
        self.visual_fc = torch.nn.Linear(self.config.hidden_size, language_hidden_size)
        self.vit_eos = torch.nn.Parameter(torch.randn(1, 1, language_hidden_size))

        self.post_init()

    def forward(
        self,
        encoder_hidden_states=None
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            batch_size is the number of all images (global+crop) in a batch
            Sequence of hidden-states at the output of the last layer of the encoder.
        """
        encoder_hidden_states = encoder_hidden_states[:,1:,:] # remove the first cls token 
        B, L, C = encoder_hidden_states.shape # B, 1024=(448/14)^2, 1024

        ## feature interaction with a conv layer
        encoder_hidden_states = rearrange(encoder_hidden_states, 'B (H W) D -> B D H W', H=int(math.sqrt(L)))
        hidden_states = self.reducer_before(encoder_hidden_states) # B 4D H W/4
        ## reduce seq length with a conv layer
        hidden_states = rearrange(hidden_states, 'B (X D) H W -> B D H (W X)', X=self.conv_patch) # B 4D H W/4 -> B D H W
        sequence_output = self.reducer(hidden_states) # B,C,H,W -> B,C,H/conv_shape[0],W/(conv_shape[1])
        sequence_output = sequence_output.flatten(2).transpose(1, 2)  # B,C,H/conv_shape[0],W/(conv_shape[1]) -> B,C,L/conv_patch -> B,L/conv_patch,C
        sequence_output = sequence_output.transpose(0, 1).contiguous() # L/conv_patch, B, C
        ## align visual features with language embedding with fc
        sequence_output = self.visual_fc(sequence_output) # L/conv_patch, B, h
        sequence_output = sequence_output.transpose(0, 1).contiguous() # B, s/4, h
        sequence_output = torch.cat([sequence_output, self.vit_eos.repeat(B, 1, 1)], dim=1)

        return sequence_output

