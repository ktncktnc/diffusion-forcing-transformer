"""
Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
"""

from typing import Tuple, Optional
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp
from ..modules.embeddings import RotaryEmbeddingND


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight


class Attention(nn.Module):
    """
    Adapted from timm.models.vision_transformer,
    to support the use of RoPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rope: Optional[RotaryEmbeddingND] = None,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        

    def forward(
        self, 
        x: torch.Tensor, 
        timestep=None,
        height: int = None,
        width: int = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        if self.fused_attn:
            # pylint: disable-next=not-callable
            x, attn_map = scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0
            )
        else:
            q = q * self.scale
            attn_map = q @ k.transpose(-2, -1)
            attn_map = attn_map.softmax(dim=-1) # shape: (B, num_heads, N, N)
            x = self.attn_drop(attn_map) @ v

        if hasattr(self, "store_attn_map"):
            assert timestep is not None, "timestep must be provided for attention map storage"
            assert height is not None and width is not None, "height and width must be provided for attention map storage"
            self.timestep = timestep
            self.attn_map = rearrange(attn_map, 'b heads (t h w) d -> b heads t h w d', h=height, w=width)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive layer norm (AdaLN).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out:
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AdaLN layer.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class AdaLayerNormZero(nn.Module):
    """
    Adaptive layer norm zero (AdaLN-Zero).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True),
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out:
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the AdaLN-Zero layer.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        shift, scale, gate = self.modulation(c).chunk(3, dim=-1)
        return modulate(self.norm(x), shift, scale), gate


class DiTBlock(nn.Module):
    """
    A DiT transformer block with adaptive layer norm zero (AdaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: Optional[float] = 4.0,
        rope: Optional[RotaryEmbeddingND] = None,
        **block_kwargs: dict,
    ):
        """
        Args:
            hidden_size: Number of features in the hidden layer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of hidden layer size in the MLP. None to skip the MLP.
            block_kwargs: Additional arguments to pass to the Attention block.
        """
        super().__init__()

        self.norm1 = AdaLayerNormZero(hidden_size)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, rope=rope, **block_kwargs
        )
        self.use_mlp = mlp_ratio is not None
        if self.use_mlp:
            self.norm2 = AdaLayerNormZero(hidden_size)
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                act_layer=partial(nn.GELU, approximate="tanh"),
            )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer linear layers:
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.attn.apply(_basic_init)
        if self.use_mlp:
            self.mlp.apply(_basic_init)

    def forward(
            self, 
            x: torch.Tensor, 
            c: torch.Tensor, 
            t: torch.Tensor=None,
            height: int = None,
            width: int = None,
        ) -> torch.Tensor:
        """
        Forward pass of the DiT block.
        In original implementation, conditioning is uniform across all tokens in the sequence. Here, we extend it to support token-wise conditioning (e.g. noise level can be different for each token).
        Args:timesteps
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        n_frames = t.shape[-1]
        
        x, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn(x, t, height, width)
        if self.use_mlp:
            x, gate_mlp = self.norm2(x, c)
            x = x + gate_mlp * self.mlp(x)
        return x


class DITFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm_final = AdaLayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out:
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Forward pass of the DiT final layer.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        """
        x = self.norm_final(x, c)
        x = self.linear(x)
        return x
