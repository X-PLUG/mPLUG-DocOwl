from typing import Union
import math
import torch
import torch.nn as nn
import re

from einops import rearrange, repeat


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class ResamplerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        image_hidden_size: int = 1024,
        num_heads: int = 12,
        intermediate_size: int = None
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "For MHSA, you must have number of heads divisible by initial hidden size"
        intermediate_size = hidden_size * 4 if intermediate_size is None else intermediate_size
        # intermediate_size = hidden_size * 4
        self.scale = 1 / math.sqrt(hidden_size // num_heads)
        self.num_heads = num_heads
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(image_hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(image_hidden_size, hidden_size, bias=False)

        self.to_out = nn.Linear(hidden_size, hidden_size, bias=False)

        self.feed_forward = nn.Sequential(
            *[
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, intermediate_size, bias=False),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size, bias=False),
            ]
        )
        # prenorm for image features
        self.norm_image = nn.LayerNorm(image_hidden_size)
        self.norm_hidden = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # prenorm
        x = self.norm_image(x)
        residual_hidden_states = hidden_states
        hidden_states = self.norm_hidden(hidden_states)
        # compute Q, K, V
        queries = self.to_q(hidden_states)
        keys = self.to_k(x)
        values = self.to_v(x)
        # rearrange them into multi-head format
        queries = rearrange(queries, "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(keys, "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(values, "b n (h d) -> b h n d", h=self.num_heads)
        # rescale
        queries = self.scale * queries
        # compute QK^T
        scores = torch.einsum("... i d, ... j d -> ... i j", queries, keys)
        # for stability
        scores = scores - scores.amax(dim=-1, keepdim=True).detach()
        # softmax
        attention_scores = scores.softmax(dim=-1)   # b h i j (i: number of queries, j: number of keys)
        # dot product with V
        out = torch.einsum("... i j, ... j d -> ... i d", attention_scores, values)
        out = rearrange(out, "b h n d -> b n (h d)", h=self.num_heads)
        out = self.to_out(out) + residual_hidden_states
        residual_out = out
        out = self.feed_forward(out)
        return out + residual_out


class Resampler(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        image_hidden_size: int = 1024,
        final_hidden_size: int = 4096,
        num_heads: int = 12,
        intermediate_size: int = None,
        num_queries: int = 128,
        num_layers: int = 3,
        initializer_range: float = 0.02
    ):
        super().__init__()
        self.resampler_blocks = nn.ModuleList(
            [
                ResamplerBlock(
                    hidden_size, image_hidden_size, num_heads, intermediate_size
                ) for _ in range(num_layers)
            ]
        )
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_size))
        self.post_norm = nn.LayerNorm(hidden_size)

        self.final_proj = nn.Linear(hidden_size, final_hidden_size, bias=False)

    #     self.initializer_range = initializer_range
    #     for module in self.modules():
    #         if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
    #             self._init_weights(module)
    #
    # def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
    #     """Initialize the weights"""
    #     if isinstance(module, (nn.Linear, nn.Conv2d)):
    #         # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
    #         # `trunc_normal_cpu` not implemented in `half` issues
    #         module.weight.data = nn.init.trunc_normal_(
    #             module.weight.data.to(torch.float32), mean=0.0, std=self.initializer_range
    #         ).to(module.weight.dtype)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        b = image_hidden_states.size(0)
        queries = repeat(self.queries, 'n d -> b n d', b=b)
        for resampler_block in self.resampler_blocks:
            queries = resampler_block(queries, image_hidden_states)

        # post norm
        queries = self.post_norm(queries)
        return self.final_proj(queries)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'resampler':
        hidden_size = getattr(config, 'resampler_hidden_size', 768)
        image_hidden_size = config.mm_hidden_size
        num_queries = getattr(config, 'num_queries', 128)
        final_hidden_size = config.hidden_size
        num_heads = 12
        if hidden_size == 512:
            num_heads = 8
        num_layers = getattr(config, 'num_resampler_layers', 3)

        initializer_range = getattr(config, 'initializer_range', 0.02)
        print(
            f"resampler config: resampler hidden size: {hidden_size}, num_queries: {num_queries}, "
            f"num_resampler_layers: {num_layers}"
        )
        return Resampler(
            hidden_size=hidden_size,
            image_hidden_size=image_hidden_size,
            num_queries=num_queries,
            final_hidden_size=final_hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            initializer_range=initializer_range
        )

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        mlp = nn.Sequential(*modules)
        if getattr(config, 'load_moe_mm_projector', False):
            from deepspeed.moe.layer import MoE
            mlp = MoE(
                config.mm_hidden_size,
                expert=mlp,
                num_experts=4,
                ep_size=1,
                k=2,
                capacity_factor=1.,
                eval_capacity_factor=1.,
                min_capacity=4,
                use_residual=False,
            )

            def moe_forward_wrapper(forward_func):
                return lambda *args, **kwargs: forward_func(*args, **kwargs)[0]
            mlp.forward = moe_forward_wrapper(mlp.forward)
        return mlp

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
