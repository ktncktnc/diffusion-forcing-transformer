"""
Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py
Extended to support:
- Temporal sequence modeling
- 1D input additionally to 2D spatial input
- Token-wise conditioning
"""

from typing import Literal, Optional, Tuple, Callable, Union, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from einops import rearrange
from ..modules.embeddings import RotaryEmbedding1D, RotaryEmbedding2D, RotaryEmbedding3D
from .dit_blocks import (
    DiTBlock,
    MatrixDiTBlock,
    MatrixSelfDiTBlock,
    MatrixCrossDiTBlock,
    DITFinalLayer,
)

Variant = Literal["full", "factorized_encoder", "factorized_attention"]
PosEmb = Literal["learned_1d", "sinusoidal_1d", "sinusoidal_2d", "sinusoidal_3d", "sinusoidal_factorized", "rope_2d", "rope_3d"]
matrix_blocks: Dict[str, nn.Module] = {
    'matrix': MatrixDiTBlock,
    'matrix_self': MatrixSelfDiTBlock,
    'matrix_cross': MatrixCrossDiTBlock,
}


def rearrange_contiguous_many(tensors: Tuple[torch.Tensor, ...], *args, **kwargs) -> Tuple[torch.Tensor, ...]:
    return tuple(rearrange(t, *args, **kwargs).contiguous() for t in tensors)

def rearrange_cont_lambda(*args, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Returns a function that rearranges a tensor and makes it contiguous if is_contiguous is True.
    """
    return lambda x, c, batch_size: rearrange_contiguous_many((x, c), *args, b=batch_size, **kwargs)

@dataclass
class ProcessingState:
    """Container for tensor processing state"""
    x: torch.Tensor
    c: torch.Tensor
    batch_size: int


class TensorProcessor:
    """Handles parallel processing of sequence and image tensors"""
    
    def __init__(self, seq_state: ProcessingState, img_state: Optional[ProcessingState] = None):
        self.seq_state = seq_state
        self.img_state = img_state
    
    def execute(self, fn: Callable[[torch.Tensor, torch.Tensor, int], Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]):
        """Execute a function in parallel on sequence and image tensors"""
        # Process sequence tensors
        seq_result = fn(self.seq_state.x, self.seq_state.c, self.seq_state.batch_size)
        self._update_state(self.seq_state, seq_result)
        
        # Process image tensors if they exist
        if self.img_state is not None:
            img_result = fn(self.img_state.x, self.img_state.c, self.img_state.batch_size)
            self._update_state(self.img_state, img_result)
    
    @staticmethod
    def _update_state(state: ProcessingState, result: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        """Update processing state with function result"""
        if isinstance(result, tuple):
            state.x, state.c = result
        else:
            state.x = result
            
class DiTBase(nn.Module):
    """
    A DiT base model.
    """
    def __init__(
        self,
        num_patches: Optional[int] = None,
        max_temporal_length: int = 16,
        out_channels: int = 4,
        variant: Variant = "full",
        pos_emb_type: PosEmb = "learned_1d",
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        use_gradient_checkpointing: bool = False,
        **kwargs
    ):
        """
        Args:
            num_patches: Number of patches in the image, None for 1D inputs.
            max_temporal_length: Maximum length of the temporal sequence.
            variant: Variant of the DiT model to use.
                - "full": process all tokens at once
                - "factorized_encoder": alternate between spatial transformer blocks and temporal transformer blocks
                - "factorized_attention": decompose the multi-head attention in the transformer block, compute spatial self-attention and then temporal self-attention
            pos_emb_type: Type of positional embedding to use.
                - "learned_1d": learned 1D positional embeddings
                - "sinusoidal_1d": sinusoidal 1D positional embeddings
                - "sinusoidal_3d": sinusoidal 3D positional embeddings
                - "sinusoidal_factorized": sinusoidal 2D positional embeddings for spatial and 1D for temporal
                - "rope_3d": rope 3D positional embeddings
        """
        super().__init__()
        self._check_args(num_patches, variant, pos_emb_type)
        self.kwargs = kwargs

        self.learn_sigma = learn_sigma
        self.out_channels = out_channels * (2 if learn_sigma else 1)
        self.num_patches = num_patches
        self.variant = variant

        self.max_temporal_length = max_temporal_length
        self.max_tokens = self.max_temporal_length * (num_patches or 1)
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        if self.is_matrix_attention:
            self.matrix_block = kwargs.get("matrix_block", None)
            self.embed_col_dim = kwargs.get("embed_col_dim", self.num_patches)
            self.embed_row_dim = kwargs.get("embed_row_dim", hidden_size)
            self.num_col_heads = self.kwargs.get("num_col_heads", num_heads)
            self.num_row_heads = self.kwargs.get("num_row_heads", num_heads)
            self.matrix_dim = self.embed_col_dim // self.num_col_heads * self.embed_row_dim // self.num_row_heads
            self.flatten_matrix_rope = kwargs.get("flatten_matrix_rope")

        self.mlp_ratio = mlp_ratio
        self.pos_emb_type = pos_emb_type
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self._build_positional_embedding()

        self.blocks = []
        for i in range(depth):
            if self.is_full_matrix:
                block = matrix_blocks[self.matrix_block]
                self.blocks.append(block(
                    col_hidden_size=self.num_patches,
                    row_hidden_size=hidden_size,
                    embed_col_dim=self.embed_col_dim,
                    embed_row_dim=self.embed_row_dim,
                    num_col_heads=self.num_col_heads,
                    num_row_heads=self.num_row_heads,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope=self.rope,
                    matrix_rope=self.temporal_rope,
                    flatten_matrix_rope=self.flatten_matrix_rope
                ))
            else:
                self.blocks.append(DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=(mlp_ratio if self.variant != "factorized_attention" else None),
                    rope=self.rope,
                ))
        self.blocks = nn.ModuleList(self.blocks)

        if self.is_factorized:
            self.temporal_blocks = []
            for i in range(depth):
                if self.is_matrix_factorized:
                    block = matrix_blocks[self.matrix_block]
                    self.temporal_blocks.append(block(
                        col_hidden_size=self.num_patches,
                        row_hidden_size=hidden_size,
                        embed_col_dim=self.embed_col_dim,
                        embed_row_dim=self.embed_row_dim,
                        num_col_heads=self.num_col_heads,
                        num_row_heads=self.num_row_heads,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        rope=self.rope,
                        matrix_rope=self.temporal_rope,
                        flatten_matrix_rope=self.flatten_matrix_rope
                    ))
                else:
                    self.temporal_blocks.append(DiTBlock(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                    ))
            self.temporal_blocks = nn.ModuleList(self.temporal_blocks)
        else:
            self.temporal_blocks = [None] * depth

        self.final_layer = DITFinalLayer(hidden_size, self.out_channels)

    def _build_positional_embedding(self):
        self.pos_emb = None
        self.spatial_pos_emb = None
        self.temporal_pos_emb = None
        self.rope = None
        self.temporal_rope = None

        match self.pos_emb_type:
            case "learned_1d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_tokens,),
                    learnable=True,
                )
            case "sinusoidal_1d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_tokens,),
                )
            case "sinusoidal_2d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.spatial_grid_size, self.spatial_grid_size),
                )
            case "sinusoidal_3d":
                self.pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(
                        self.max_temporal_length,
                        self.spatial_grid_size,
                        self.spatial_grid_size,
                    ),
                )
            case "sinusoidal_factorized":
                self.spatial_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.spatial_grid_size, self.spatial_grid_size),
                )
                self.temporal_pos_emb = SinusoidalPositionalEmbedding(
                    embed_dim=self.hidden_size,
                    shape=(self.max_temporal_length,),
                )
            case "rope_2d":
                self.rope = RotaryEmbedding2D(
                        dim=self.hidden_size//self.num_heads,
                        sizes=(
                            self.spatial_grid_size,
                            self.spatial_grid_size,
                        ),
                )
            case "rope_3d":
                self.rope = RotaryEmbedding3D(
                        dim=self.hidden_size//self.num_heads,
                        sizes=(
                            self.max_temporal_length,
                            self.spatial_grid_size,
                            self.spatial_grid_size,
                        ),
                    )
        
        if self.is_matrix_attention:
            # NOTE: RoPE will be added for each row of the matrix independently. Doing this makes sure every row has its own frequencies. If we flatten the row and column, frequencies will be smaller by row.
            self.temporal_rope = RotaryEmbedding1D(
                    dim=self.embed_row_dim // self.num_row_heads,
                    seq_len=self.max_temporal_length
                )
    
    def forward(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor, 
        t: torch.Tensor = None,
        height: int = None,
        width: int = None
    ) -> torch.Tensor:
        """
        Forward pass of the DiTBase model.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
        Returns:
            Output tensor of shape (B, N, OC).
        """
        x_img = None
        if x.size(1) > self.max_tokens:
            if not self.training or self.num_patches is None:
                raise ValueError(f"Input sequence length {x.size(1)} exceeds the maximum length {self.max_tokens}")

            else:  # image-video joint training
                video_end = self.max_temporal_length * self.num_patches
                x, x_img, c, c_img = (x[:, :video_end], x[:, video_end:], c[:, :video_end], c[:, video_end:])

                x_img, c_img = rearrange_contiguous_many(
                    (x_img, c_img), "b (t p) c -> (b t) p c", p=self.num_patches
                )  # as if they are sequences of length 1

        seq_batch_size = x.size(0)
        img_batch_size = x_img.size(0) if x_img is not None else None

        seq_states = ProcessingState(x=x, c=c, batch_size=seq_batch_size)
        img_states = ProcessingState(x=x_img, c=c_img, batch_size=img_batch_size) if x_img is not None else None
        processor = TensorProcessor(seq_states, img_states)

        #---------------------------------------
        # Absolute positional embeddings
        #---------------------------------------
        # learned_1d, sinusoidal_1d, sinusoidal_3d
        # 1d means flattened sequence
        # 3d means temporal + spatial
        if self.is_pos_emb_absolute_once:
            processor.execute(lambda x, c, batch_size: self.pos_emb(x))
        
        # sinusoidal_2d: only for spatial, temporal will be added using rope1d
        if self.is_pos_emb_absolute_2d_once:
            def add_pos_emb(x: torch.Tensor, _: torch.Tensor, batch_size: int) -> torch.Tensor:
                x = rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
                x = self.pos_emb(x)
                x = rearrange(x, "(b t) p c -> b (t p) c", b=batch_size)
                return x
            processor.execute(add_pos_emb)

        # factorized sinusoidal 3D: spatial + temporal
        if self.is_pos_emb_absolute_factorized:
            # temporal PE will be added later
            if self.is_factorized:
                def add_pos_emb(x: torch.Tensor, _: torch.Tensor, batch_size: int) -> torch.Tensor:
                    x = rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
                    x = self.spatial_pos_emb(x)
                    x = rearrange(x, "(b t) p c -> b (t p) c", b=batch_size)
                    return x
            # full PE for 3D input
            else:
                def add_pos_emb(x: torch.Tensor, _: torch.Tensor, batch_size: int) -> torch.Tensor:
                    x = rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
                    x = self.spatial_pos_emb(x)
                    x = rearrange(x, "(b t) p c -> (b p) t c", b=batch_size)
                    x = self.temporal_pos_emb(x)
                    x = rearrange(x, "(b p) t c -> b (t p) c", b=batch_size)
                    return x

            processor.execute(add_pos_emb)
        #----------------------------------------
        # End of absolute positional embeddings
        #----------------------------------------

        # factorized sinusoidal_factorized
        if self.is_factorized:
            processor.execute(rearrange_cont_lambda("b (t p) c -> (b t) p c", p=self.num_patches))

        for i, (block, temporal_block) in enumerate(zip(self.blocks, self.temporal_blocks)):
            # spatial block or full block
            processor.execute(lambda x, c, batch_size: self.checkpoint(block, x, c, t, height, width))

            # Factorized matrix attention
            if self.is_matrix_factorized:
                #processor.execute(lambda x, c, batch_size: rearrange_contiguous_many((x, c), "(b t) p c -> b (p t) c", b=batch_size))
                processor.execute(rearrange_cont_lambda("(b t) p c -> b (p t) c"))
                processor.execute(lambda x, c, batch_size: self.checkpoint(temporal_block, x, c, t, height, width))
                processor.execute(rearrange_cont_lambda("b (p t) c -> (b t) p c", p=self.num_patches))

            elif self.is_factorized:
                processor.execute(rearrange_cont_lambda("(b t) p c -> (b p) t c"))
                if i == 0 and self.pos_emb_type == "sinusoidal_factorized":
                    processor.execute(lambda x, c, batch_size: self.temporal_pos_emb(x))

                processor.execute(lambda x, c, batch_size: self.checkpoint(temporal_block, x, c, t, height, width))
                processor.execute(rearrange_cont_lambda("(b p) t c -> (b t) p c"))

        if self.is_factorized:
            processor.execute(lambda x, c, batch_size: rearrange_contiguous_many((x, c), "(b t) p c -> b (t p) c", b=batch_size))

        processor.execute(lambda x, c, batch_size: self.final_layer(x, c))

        x = seq_states.x
        if img_states is not None:
            x_img = rearrange(img_states.x, "(b t) p c -> b (t p) c", b=seq_batch_size)
            x = torch.cat([x, x_img], dim=1)
        return x

    @property
    def is_factorized(self) -> bool:
        return self.variant in {"factorized_encoder", "factorized_attention", "factorized_matrix_attention"}
    
    @property
    def is_matrix_factorized(self) -> bool:
        return self.variant == "factorized_matrix_attention"
    
    @property
    def is_full_matrix(self) -> bool:
        return self.variant == 'full_matrix_attention'
    
    @property
    def is_matrix_attention(self) -> bool:
        return self.variant in {"full_matrix_attention", "factorized_matrix_attention"}

    @property
    def is_pos_emb_absolute_once(self) -> bool:
        return self.pos_emb_type in {"learned_1d", "sinusoidal_1d", "sinusoidal_3d"}
    
    @property
    def is_pos_emb_absolute_2d_once(self) -> bool:
        return self.pos_emb_type in {"sinusoidal_2d"}

    @property
    def is_pos_emb_absolute_factorized(self) -> bool:
        return self.pos_emb_type == "sinusoidal_factorized"

    @property
    def spatial_grid_size(self) -> Optional[int]:
        if self.num_patches is None:
            return None
        grid_size = int(self.num_patches**0.5)
        assert (
            grid_size * grid_size == self.num_patches
        ), "num_patches must be a square number"
        return grid_size

    @staticmethod
    def _check_args(num_patches: Optional[int], variant: Variant, pos_emb_type: PosEmb):
        if variant not in {"full", "factorized_encoder", "factorized_attention", "factorized_matrix_attention", "full_matrix_attention"}:
            raise ValueError(f"Unknown variant {variant}")
        if pos_emb_type not in {
            "learned_1d",
            "sinusoidal_1d",
            "sinusoidal_2d",
            "sinusoidal_3d",
            "sinusoidal_factorized",
            "rope_3d",
            "rope_2d",
            "rope_1d"
        }:
            raise ValueError(f"Unknown positional embedding type {pos_emb_type}")
        if num_patches is None:
            assert (
                variant == "full"
            ), "For 1D inputs, factorized variants are not supported"
            assert pos_emb_type in {
                "learned_1d",
                "sinusoidal_1d",
            }, "For 1D inputs, only 1D positional embeddings are supported"

        if pos_emb_type == "rope_3d":
            assert variant == "full", "Rope3D is only supported with full variant"

    def checkpoint(self, module: nn.Module, *args):
        if self.use_gradient_checkpointing:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, shape: Tuple[int, ...], learnable: bool = False):
        super().__init__()
        if learnable:
            max_tokens = np.prod(shape)
            self.pos_emb = nn.Parameter(
                torch.zeros(1, max_tokens, embed_dim).normal_(std=0.02),
                requires_grad=True,
            )

        else:
            self.register_buffer(
                "pos_emb",
                torch.from_numpy(get_nd_sincos_pos_embed(embed_dim, shape))
                .float()
                .unsqueeze(0),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pos_emb[:, :seq_len]


def get_nd_sincos_pos_embed(
    embed_dim: int,
    shape: Tuple[int, ...],
) -> np.ndarray:
    """
    Get n-dimensional sinusoidal positional embeddings.
    Args:
        embed_dim: Embedding dimension.
        shape: Shape of the input tensor.
    Returns:
        Positional embeddings with shape (shape_flattened, embed_dim).
    """
    assert embed_dim % (2 * len(shape)) == 0
    grid = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape])
    grid = np.stack(grid, axis=0)  # (ndim, *shape)
    return np.concatenate(
        [
            get_1d_sincos_pos_embed_from_grid(embed_dim // len(shape), grid[i])
            for i in range(len(shape))
        ],
        axis=1,
    )


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Args:
        embed_dim: Embedding dimension.
        pos: Position tensor of shape (...).
    Returns:
        Positional embeddings with shape (-1, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
