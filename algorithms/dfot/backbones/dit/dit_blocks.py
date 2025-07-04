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
            .permute(2, 0, 3, 1, 4) # (3, batch_size, num_heads, num_tokens, head_dim)
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


class CrossAttention(nn.Module):
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

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        

    def forward(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        timestep=None,
        height: int = None,
        width: int = None
    ) -> torch.Tensor:
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, num_heads, N, head_dim)
        kv = (
            self.kv_proj(y)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4) # (2, batch_size, num_heads, num_tokens, head_dim)
        )
        k, v = kv.unbind(0)
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
    

def matrix_mul(x, u, v):
    return torch.einsum('nm,blnd,dk->blmk', u, x, v)


class MatrixAttention(nn.Module):
    """
    Matrix attention block.
    This is a simplified version of the attention block that does not use RoPE.
    It is used in the DiT model for the final layer.
    """
    def __init__(
        self, 
        col_dim: int,
        row_dim: int,
        embed_col_dim: Optional[int] = None,
        embed_row_dim: Optional[int] = None, 
        num_col_heads: int = 4,
        num_row_heads: int = 4,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rope = None,
        **kwargs
        # fused_attn: bool = True,
    ):
        super().__init__()
        self.col_dim = col_dim
        self.row_dim = row_dim
        self.embed_col_dim = embed_col_dim if embed_col_dim is not None else col_dim
        self.embed_row_dim = embed_row_dim if embed_row_dim is not None else row_dim
        self.num_col_heads = num_col_heads
        self.num_row_heads = num_row_heads
        self.num_heads = num_col_heads * num_row_heads
        assert self.embed_col_dim % num_col_heads == 0, "embed_col_dim must be divisible by num_col_heads"
        assert self.embed_row_dim % num_row_heads == 0, "embed_row_dim must be divisible by num_row_heads"
        self.head_col_dim = self.embed_col_dim // num_col_heads
        self.head_row_dim = self.embed_row_dim // num_row_heads
        self.scale = (self.head_col_dim*self.head_row_dim)**-0.5

        self.qkv_u = nn.Parameter(torch.rand(col_dim, self.embed_col_dim))
        self.qkv_v = nn.Parameter(torch.rand(row_dim, self.embed_row_dim*3))
        self.q_norm = norm_layer([self.head_col_dim, self.head_row_dim]) if qk_norm else nn.Identity()
        self.k_norm = norm_layer([self.head_col_dim, self.head_row_dim]) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_u = nn.Parameter(torch.rand(self.embed_col_dim, col_dim))
        self.proj_v = nn.Parameter(torch.rand(self.embed_row_dim, row_dim))
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(
        self, 
        x: torch.Tensor, 
        timestep=None,
        height: int = None,
        width: int = None
    ) -> torch.Tensor:
        B, L, H, W = x.shape
        qkv = (
            matrix_mul(x, self.qkv_u, self.qkv_v) # BLH*3,W
            .reshape(B, L, 3, self.num_col_heads, self.head_col_dim, self.num_row_heads, self.head_row_dim)
            .permute(2, 0, 3, 5, 1, 4, 6) # (3. B, num_col_heads, num_row_heads, N, H_e, W_e)
        )
        q, k, v = qkv.unbind(0) # batch_size, num_col_heads, num_row_heads, n_tokens, height, width
        q, k = self.q_norm(q), self.k_norm(k) 

        # Currently, only support RoPE1D, add positional embeddings for each frame.
        if self.rope is not None:
            q = self.rope(q.flatten(-2,-1)).reshape(B, self.num_col_heads, self.num_row_heads, L, self.head_col_dim, self.head_row_dim)
            k = self.rope(k.flatten(-2,-1)).reshape(B, self.num_col_heads, self.num_row_heads, L, self.head_col_dim, self.head_row_dim)

        attn_map = torch.einsum('bmnihw,bmnjhw->bmnij', q, k)  # (B, num_heads, N, N)
        attn_map = (attn_map/self.scale).softmax(dim=-1)  # shape: (B, num_heads, N, N)

        x = torch.einsum('bmnij,bmnjhw->bmnihw', self.attn_drop(attn_map), v) # (B, col_num_head, row_num_head, num_tokens, H, W)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, N, num_heads, H, W)
        x = x.reshape(B, L, self.num_col_heads * self.head_col_dim, self.num_row_heads * self.head_row_dim)
        x = matrix_mul(x, self.proj_u, self.proj_v)  # (B, N, C)
        x = self.proj_drop(x)
        if hasattr(self, "store_attn_map"):
            raise NotImplementedError("MatrixAttention does not support storing attention maps.")

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


# ---------------------------------------------------------------------
# Matrix DiT
# ---------------------------------------------------------------------

class MatrixDiTBlock(nn.Module):
    """
    A MatrixDiT transformer block with adaptive layer norm zero (AdaLN-Zero) conditioning.
    """

    def __init__(
        self,
        col_hidden_size: int,
        row_hidden_size: int,
        num_col_heads: int,
        num_row_heads: int,
        embed_col_dim: Optional[int] = None,
        embed_row_dim: Optional[int] = None,
        mlp_ratio: Optional[float] = 4.0,
        matrix_rope: Optional[RotaryEmbeddingND] = None,
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

        self.norm1 = AdaLayerNormZero(row_hidden_size)
        self.attn = MatrixAttention(
            col_dim=col_hidden_size,
            row_dim=row_hidden_size,
            embed_col_dim=embed_col_dim,
            embed_row_dim=embed_row_dim,
            num_col_heads=num_col_heads,
            num_row_heads=num_row_heads,
            rope=matrix_rope, 
            **block_kwargs
        )
        self.use_mlp = mlp_ratio is not None
        if self.use_mlp:
            self.norm2 = AdaLayerNormZero(row_hidden_size)
            self.mlp = Mlp(
                in_features=row_hidden_size,
                hidden_features=int(row_hidden_size * mlp_ratio),
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
            elif isinstance(module, MatrixAttention):
                # Initialize the projection matrices in MatrixAttention
                nn.init.xavier_uniform_(module.qkv_u)
                nn.init.xavier_uniform_(module.qkv_v)
                nn.init.xavier_uniform_(module.proj_u)
                nn.init.xavier_uniform_(module.proj_v)
            
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
        B, N, C = x.shape
        n_frames = t.shape[-1]
        x, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn(x.reshape(B, n_frames, N//n_frames, C), t, height, width).reshape(B, N, C)
        if self.use_mlp:
            x, gate_mlp = self.norm2(x, c)
            x = x + gate_mlp * self.mlp(x)
        return x
    

class MatrixCrossDiTBlock(nn.Module):
    """
    A MatrixDiT transformer block with adaptive layer norm zero (AdaLN-Zero) conditioning.
    """

    def __init__(
        self,
        col_hidden_size: int,
        row_hidden_size: int,
        num_col_heads: int,
        num_row_heads: int,
        embed_col_dim: Optional[int] = None,
        embed_row_dim: Optional[int] = None,
        mlp_ratio: Optional[float] = 4.0,
        rope: Optional[RotaryEmbeddingND] = None,
        matrix_rope: Optional[RotaryEmbeddingND] = None,
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

        self.norm1 = AdaLayerNormZero(row_hidden_size)
        self.attn1 = MatrixAttention(
            col_dim=col_hidden_size,
            row_dim=row_hidden_size,
            embed_col_dim=embed_col_dim,
            embed_row_dim=embed_row_dim,
            num_col_heads=num_col_heads,
            num_row_heads=num_row_heads,
            rope=matrix_rope, 
        )
        self.attn2 = CrossAttention(
            dim=row_hidden_size,
            num_heads=num_row_heads,
            qkv_bias=True,
            rope=rope,
        )
        self.use_mlp = mlp_ratio is not None
        if self.use_mlp:
            self.norm3 = AdaLayerNormZero(row_hidden_size)
            self.mlp = Mlp(
                in_features=row_hidden_size,
                hidden_features=int(row_hidden_size * mlp_ratio),
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
            elif isinstance(module, MatrixAttention):
                # Initialize the projection matrices in MatrixAttention
                nn.init.xavier_uniform_(module.qkv_u)
                nn.init.xavier_uniform_(module.qkv_v)
                nn.init.xavier_uniform_(module.proj_u)
                nn.init.xavier_uniform_(module.proj_v)

        self.attn1.apply(_basic_init)
        self.attn2.apply(_basic_init)
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
        B, N, C = x.shape
        n_frames = t.shape[-1]
        
        # Temporal matrix attention
        x, gate_msa = self.norm1(x, c)
        x1 = rearrange(x, 'b (t p) c -> b t p c', t=n_frames)
        x1 =  self.attn1(x1, t, height, width)   
        attn_x = self.attn2(
            rearrange(x, 'b (t p) c -> (b t) p c', t=n_frames), 
            rearrange(x1, 'b t p c -> (b t) p c', t=n_frames), 
            t, 
            height, 
            width
        )
        x = x + gate_msa * rearrange(attn_x, '(b t) p c -> b (t p) c', t=n_frames)

        # MLP
        if self.use_mlp:
            x, gate_mlp = self.norm3(x, c)
            x = x + gate_mlp * self.mlp(x)
        return x


class MatrixSelfDiTBlock(nn.Module):
    """
    A MatrixDiT transformer block with adaptive layer norm zero (AdaLN-Zero) conditioning.
    """

    def __init__(
        self,
        col_hidden_size: int,
        row_hidden_size: int,
        num_col_heads: int,
        num_row_heads: int,
        embed_col_dim: Optional[int] = None,
        embed_row_dim: Optional[int] = None,
        mlp_ratio: Optional[float] = 4.0,
        rope: Optional[RotaryEmbeddingND] = None,
        matrix_rope: Optional[RotaryEmbeddingND] = None,
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

        self.norm1 = AdaLayerNormZero(row_hidden_size)
        self.attn1 = MatrixAttention(
            col_dim=col_hidden_size,
            row_dim=row_hidden_size,
            embed_col_dim=embed_col_dim,
            embed_row_dim=embed_row_dim,
            num_col_heads=num_col_heads,
            num_row_heads=num_row_heads,
            rope=matrix_rope, 
        )
        self.norm2 = AdaLayerNormZero(row_hidden_size)
        self.attn2 = Attention(
            dim=row_hidden_size,
            num_heads=num_row_heads,
            qkv_bias=True,
            rope=rope
        )
        self.use_mlp = mlp_ratio is not None
        if self.use_mlp:
            self.norm3 = AdaLayerNormZero(row_hidden_size)
            self.mlp = Mlp(
                in_features=row_hidden_size,
                hidden_features=int(row_hidden_size * mlp_ratio),
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
            elif isinstance(module, MatrixAttention):
                # Initialize the projection matrices in MatrixAttention
                nn.init.xavier_uniform_(module.qkv_u)
                nn.init.xavier_uniform_(module.qkv_v)
                nn.init.xavier_uniform_(module.proj_u)
                nn.init.xavier_uniform_(module.proj_v)

        self.attn1.apply(_basic_init)
        self.attn2.apply(_basic_init)
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
        B, N, C = x.shape
        n_frames = t.shape[-1]
        
        # Temporal matrix attention
        x, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn1(rearrange(x, 'b (t p) c -> b t p c', t=n_frames), t, height, width).reshape(B, N, C)

        # Self attention
        x, gate_msa = self.norm2(x, c)
        attn_x = self.attn2(rearrange(x, 'b (t p) c -> (b t) p c', t=n_frames), t, height, width)
        x = x + gate_msa * rearrange(attn_x, '(b t) p c -> b (t p) c', t=n_frames)

        # MLP
        if self.use_mlp:
            x, gate_mlp = self.norm3(x, c)
            x = x + gate_mlp * self.mlp(x)
        return x
    

if __name__ == "__main__":
    # Test matrix DiT block
    batch_size, n_frames, n_tokens, token_dim = 5, 16, 16*16, 768
    from algorithms.dfot.backbones.modules.embeddings import RotaryEmbedding1D
    
    rope = RotaryEmbedding1D(
        128*128, 16
    )
    block = MatrixDiTBlock(
        col_hidden_size=n_frames,
        row_hidden_size=token_dim,
        num_col_heads=4,
        num_row_heads=4,
        embed_col_dim=512,
        embed_row_dim=512,
        mlp_ratio=4.0,
        rope=rope,
    )

    x = torch.randn(batch_size, n_tokens, token_dim)
    print("Input shape:", x.shape)  # Should be (batch_size, n_tokens, token_dim)
    out = block(
        x,
        c=torch.randn(batch_size, n_tokens, token_dim),
        t=torch.randint(0, 1000, (batch_size, n_frames))
    )
    print("Output shape:", out.shape)  # Should be (batch_size, n_tokens, token_dim)
    print('output isnan:', torch.isnan(out).any().item())
    # NLP Example
    
    # x = torch.arange(1, 97).reshape(1, 2, 8, 6) # B=2, T=2, D=8 
    # # col_head: 2
    # # row_head: 3
    # print(x)
    # # x = x.reshape(1, 2, 2, 4, 6).permute(0, 2, 1, 3, 4) # (B, num_heads, N, H_e, W),  (1, 2, 2, 4, 6)
    # # x = x.reshape(1, 2, 2, 4, 3, 2).permute(0, 1, 4, 2, 3, 5)
    # x = x.reshape(1, 2, 2, 4, 3, 2).permute(0, 2, 4, 1, 3, 5)

    # print(x) # (1, 2, 3, 2, 4, 2) # (B, H_num_heads, W_num_heads, N, H_e, W), 
    # # print(x.shape)


    # x = x.permute(0, 3, 1, 4, 2, 5) # (B, N, H_num_heads, H_e, W_num_heads, W_e)
    # x = x.reshape(1, 2, 8, 6)
    # print(x)

    # # B, N, C = x.shape # (B, N, C)
    # # qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4).contiguous() # (B, N, 3, H, D)
    # # q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) # (B, H, N, D)
        
    # qkv = (
    #     matrix_mul(x, self.qkv_u, self.qkv_v) # BLH*3,W
    #     .reshape(B, L, 3, self.num_col_heads, self.head_col_dim, self.num_row_heads, self.head_row_dim)
    #     .permute(2, 0, 3, 5, 1, 4, 6)
    # )
    #q, k, v = qkv.unbind(0) # batch_size, n_heads, n_tokens, height, width
    # q, k = self.q_norm(q), self.k_norm(k) 

    # # Currently, only support RoPE1D, add positional embeddings for each frame.
    # # I think the matrix is already support position of height, width inside a frame.
    # # TODO: check if this is correct.
    # # if self.rope is not None:
    # #     q = self.rope(q.flatten(-2,-1)).reshape(B, self.num_heads, L, self.head_col_dim, self.head_row_dim)
    # #     k = self.rope(k.flatten(-2,-1)).reshape(B, self.num_heads, L, self.head_col_dim, self.head_row_dim)

    # attn_map = torch.einsum('bmnihw,bmnjhw->bmnij', q, k)  # (B, num_heads, N, N)
    # attn_map = (attn_map/self.scale).softmax(dim=-1)  # shape: (B, num_heads, N, N)

    # x = torch.einsum('bmnij,bmnjhw->bmnihw', self.attn_drop(attn_map), v) # (B, col_num_head, row_num_head, num_tokens, H, W)
    # x = x.permute(0, 3, 1, 4, 2, 5)  # (B, N, num_heads, H, W)
    # x = x.reshape(B, L, self.num_col_heads * self.head_col_dim, self.num_row_heads * self.head_row_dim)

    # B, L, H, W = 1, 3, 8, 6
    # n_H, n_W = 2, 3  # Number of heads in height and width dimensions
    # X = (torch.arange(1, 3*B*L*H*W+1) 
    # .reshape(B, L, 3, n_H, H//n_H, n_W, W//n_W)  # (B, L, H, W)
    # .permute(2, 0, 3, 5, 1, 4, 6)  # (H, B, n_H, n_W, L, H_e, W_e)
    # )
    # q,k,v = X.unbind(0)  # (B, n_H, n_W, L, H_e, W_e)

    # print(v)

    # v = v.permute(0, 3, 1, 4, 2, 5)  # (B, L, n_H, H_e, n_W, W_e)
    # v = v.reshape(B, L, H, W)
    # print(v)



    
