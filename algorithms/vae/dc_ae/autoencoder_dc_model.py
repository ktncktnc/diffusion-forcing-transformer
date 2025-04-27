# Copyright 2024 MIT, Tsinghua University, NVIDIA CORPORATION and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from omegaconf import DictConfig
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pathlib import Path
from typing import Optional, Tuple, Union
from safetensors.torch import load_file
from omegaconf import OmegaConf
from einops import rearrange
from utils.storage_utils import safe_torch_save
from utils.logging_utils import log_video
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import SanaMultiscaleAttentionProjection, SanaMultiscaleAttnProcessor2_0
from diffusers.models.autoencoders.vae import DecoderOutput, EncoderOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm, get_normalization
from diffusers.models.transformers.sana_transformer import GLUMBConv
from diffusers.utils.accelerate_utils import apply_forward_hook
from utils.ckpt_utils import (
    is_wandb_run_path,
    is_hf_path,
    wandb_to_local_path,
    download_pretrained as hf_to_local_path,
)
# from far.utils.registry import MODEL_REGISTRY


class SanaMultiscaleLinearAttention(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_attention_heads: Optional[int] = None,
        attention_head_dim: int = 8,
        mult: float = 1.0,
        norm_type: str = 'batch_norm',
        kernel_sizes: Tuple[int, ...] = (5, ),
        eps: float = 1e-15,
        residual_connection: bool = False,
    ):
        super().__init__()

        # To prevent circular import

        self.eps = eps
        self.attention_head_dim = attention_head_dim
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        num_attention_heads = (int(in_channels // attention_head_dim * mult) if num_attention_heads is None else num_attention_heads)
        inner_dim = num_attention_heads * attention_head_dim

        self.to_q = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_k = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_v = nn.Linear(in_channels, inner_dim, bias=False)

        self.to_qkv_multiscale = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.to_qkv_multiscale.append(SanaMultiscaleAttentionProjection(inner_dim, num_attention_heads, kernel_size))

        self.nonlinearity = nn.ReLU()
        self.to_out = nn.Linear(inner_dim * (1 + len(kernel_sizes)), out_channels, bias=False)
        self.norm_out = get_normalization(norm_type, num_features=out_channels)

        self.processor = SanaMultiscaleAttnProcessor2_0()

    def apply_linear_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        value = F.pad(value, (0, 0, 0, 1), mode='constant', value=1)  # Adds padding
        with torch.cuda.amp.autocast(enabled=False):
            scores = torch.matmul(value, key.transpose(-1, -2))
            hidden_states = torch.matmul(scores, query)

        hidden_states = hidden_states.to(dtype=torch.float32)
        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)
        return hidden_states

    def apply_quadratic_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(key.transpose(-1, -2), query)
        scores = scores.to(dtype=torch.float32)
        scores = scores / (torch.sum(scores, dim=2, keepdim=True) + self.eps)
        hidden_states = torch.matmul(value.to(scores.dtype), scores).to(value.dtype)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states)


class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = 'batch_norm',
        act_fn: str = 'relu6',
    ) -> None:
        super().__init__()

        self.norm_type = norm_type

        self.nonlinearity = get_activation(act_fn) if act_fn is not None else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm = get_normalization(norm_type, out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.norm_type == 'rms_norm':
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states + residual


class EfficientViTBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        mult: float = 1.0,
        attention_head_dim: int = 32,
        qkv_multiscales: Tuple[int, ...] = (5, ),
        norm_type: str = 'batch_norm',
    ) -> None:
        super().__init__()

        self.attn = SanaMultiscaleLinearAttention(
            in_channels=in_channels,
            out_channels=in_channels,
            mult=mult,
            attention_head_dim=attention_head_dim,
            norm_type=norm_type,
            kernel_sizes=qkv_multiscales,
            residual_connection=True,
        )

        self.conv_out = GLUMBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            norm_type='rms_norm',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.conv_out(x)
        return x


def get_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    attention_head_dim: int,
    norm_type: str,
    act_fn: str,
    qkv_mutliscales: Tuple[int] = (),
):
    if block_type == 'ResBlock':
        block = ResBlock(in_channels, out_channels, norm_type, act_fn)

    elif block_type == 'EfficientViTBlock':
        block = EfficientViTBlock(in_channels, attention_head_dim=attention_head_dim, norm_type=norm_type, qkv_multiscales=qkv_mutliscales)

    else:
        raise ValueError(f'Block with {block_type=} is not supported.')

    return block


class DCDownBlock2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        shortcut: bool = True
    ) -> None:
        super().__init__()

        self.downsample = downsample
        self.factor = 2
        self.stride = 1 if downsample else 2
        self.group_size = in_channels * self.factor**2 // out_channels
        self.shortcut = shortcut

        out_ratio = self.factor**2
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.conv(hidden_states)

        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)

        if self.shortcut:
            y = F.pixel_unshuffle(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class DCUpBlock2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = 'nearest',
    ) -> None:
        super().__init__()

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor**2 // in_channels

        out_ratio = self.factor**2

        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        attention_head_dim: int = 32,
        block_type: Union[str, Tuple[str]] = 'ResBlock',
        block_out_channels: Tuple[int] = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: Tuple[int] = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5, ), (5, ), (5, )),
        downsample_block_type: str = 'pixel_unshuffle',
        out_shortcut: bool = True,
    ):
        super().__init__()

        num_blocks = len(block_out_channels)

        if isinstance(block_type, str):
            block_type = (block_type, ) * num_blocks

        if layers_per_block[0] > 0:
            self.conv_in = nn.Conv2d(
                in_channels,
                block_out_channels[0]
                if layers_per_block[0] > 0 else block_out_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv_in = DCDownBlock2d(
                in_channels=in_channels,
                out_channels=block_out_channels[0]
                if layers_per_block[0] > 0 else block_out_channels[1],
                downsample=downsample_block_type == 'pixel_unshuffle',
                shortcut=False,
            )

        down_blocks = []
        for i, (out_channel, num_layers) in enumerate(zip(block_out_channels, layers_per_block)):
            down_block_list = []

            for _ in range(num_layers):
                block = get_block(
                    block_type[i],
                    out_channel,
                    out_channel,
                    attention_head_dim=attention_head_dim,
                    norm_type='rms_norm',
                    act_fn='silu',
                    qkv_mutliscales=qkv_multiscales[i],
                )
                down_block_list.append(block)

            if i < num_blocks - 1 and num_layers > 0:
                downsample_block = DCDownBlock2d(
                    in_channels=out_channel,
                    out_channels=block_out_channels[i + 1],
                    downsample=downsample_block_type == 'pixel_unshuffle',
                    shortcut=True,
                )
                down_block_list.append(downsample_block)

            down_blocks.append(nn.Sequential(*down_block_list))

        self.down_blocks = nn.ModuleList(down_blocks)

        self.conv_out = nn.Conv2d(block_out_channels[-1], latent_channels, 3, 1, 1)

        self.out_shortcut = out_shortcut
        if out_shortcut:
            self.out_shortcut_average_group_size = block_out_channels[-1] // latent_channels

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        if self.out_shortcut:
            x = hidden_states.unflatten(1, (-1, self.out_shortcut_average_group_size))
            x = x.mean(dim=2)
            hidden_states = self.conv_out(hidden_states) + x
        else:
            hidden_states = self.conv_out(hidden_states)

        return hidden_states


class Decoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        attention_head_dim: int = 32,
        block_type: Union[str, Tuple[str]] = 'ResBlock',
        block_out_channels: Tuple[int] = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: Tuple[int] = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5, ), (5, ), (5, )),
        norm_type: Union[str, Tuple[str]] = 'rms_norm',
        act_fn: Union[str, Tuple[str]] = 'silu',
        upsample_block_type: str = 'pixel_shuffle',
        in_shortcut: bool = True,
    ):
        super().__init__()

        num_blocks = len(block_out_channels)

        if isinstance(block_type, str):
            block_type = (block_type, ) * num_blocks
        if isinstance(norm_type, str):
            norm_type = (norm_type, ) * num_blocks
        if isinstance(act_fn, str):
            act_fn = (act_fn, ) * num_blocks

        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[-1], 3, 1, 1)

        self.in_shortcut = in_shortcut
        if in_shortcut:
            self.in_shortcut_repeats = block_out_channels[-1] // latent_channels

        up_blocks = []
        for i, (out_channel, num_layers) in reversed(list(enumerate(zip(block_out_channels, layers_per_block)))):
            up_block_list = []

            if i < num_blocks - 1 and num_layers > 0:
                upsample_block = DCUpBlock2d(
                    block_out_channels[i + 1],
                    out_channel,
                    interpolate=upsample_block_type == 'interpolate',
                    shortcut=True,
                )
                up_block_list.append(upsample_block)

            for _ in range(num_layers):
                block = get_block(
                    block_type[i],
                    out_channel,
                    out_channel,
                    attention_head_dim=attention_head_dim,
                    norm_type=norm_type[i],
                    act_fn=act_fn[i],
                    qkv_mutliscales=qkv_multiscales[i],
                )
                up_block_list.append(block)

            up_blocks.insert(0, nn.Sequential(*up_block_list))

        self.up_blocks = nn.ModuleList(up_blocks)

        channels = block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1]

        self.norm_out = RMSNorm(channels, 1e-5, elementwise_affine=True, bias=True)
        self.conv_act = nn.ReLU()
        self.conv_out = None

        if layers_per_block[0] > 0:
            self.conv_out = nn.Conv2d(channels, in_channels, 3, 1, 1)
        else:
            self.conv_out = DCUpBlock2d(
                channels,
                in_channels,
                interpolate=upsample_block_type == 'interpolate',
                shortcut=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.in_shortcut:
            x = hidden_states.repeat_interleave(self.in_shortcut_repeats, dim=1)
            hidden_states = self.conv_in(hidden_states) + x
        else:
            hidden_states = self.conv_in(hidden_states)

        for up_block in reversed(self.up_blocks):
            hidden_states = up_block(hidden_states)

        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


# @MODEL_REGISTRY.register()
class MyAutoencoderDC(ModelMixin, ConfigMixin):
    r"""
    An Autoencoder model introduced in [DCAE](https://arxiv.org/abs/2410.10733) and used in
    [SANA](https://arxiv.org/abs/2410.10629).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Args:
        in_channels (`int`, defaults to `3`):
            The number of input channels in samples.
        latent_channels (`int`, defaults to `32`):
            The number of channels in the latent space representation.
        encoder_block_types (`Union[str, Tuple[str]]`, defaults to `"ResBlock"`):
            The type(s) of block to use in the encoder.
        decoder_block_types (`Union[str, Tuple[str]]`, defaults to `"ResBlock"`):
            The type(s) of block to use in the decoder.
        encoder_block_out_channels (`Tuple[int, ...]`, defaults to `(128, 256, 512, 512, 1024, 1024)`):
            The number of output channels for each block in the encoder.
        decoder_block_out_channels (`Tuple[int, ...]`, defaults to `(128, 256, 512, 512, 1024, 1024)`):
            The number of output channels for each block in the decoder.
        encoder_layers_per_block (`Tuple[int]`, defaults to `(2, 2, 2, 3, 3, 3)`):
            The number of layers per block in the encoder.
        decoder_layers_per_block (`Tuple[int]`, defaults to `(3, 3, 3, 3, 3, 3)`):
            The number of layers per block in the decoder.
        encoder_qkv_multiscales (`Tuple[Tuple[int, ...], ...]`, defaults to `((), (), (), (5,), (5,), (5,))`):
            Multi-scale configurations for the encoder's QKV (query-key-value) transformations.
        decoder_qkv_multiscales (`Tuple[Tuple[int, ...], ...]`, defaults to `((), (), (), (5,), (5,), (5,))`):
            Multi-scale configurations for the decoder's QKV (query-key-value) transformations.
        upsample_block_type (`str`, defaults to `"pixel_shuffle"`):
            The type of block to use for upsampling in the decoder.
        downsample_block_type (`str`, defaults to `"pixel_unshuffle"`):
            The type of block to use for downsampling in the encoder.
        decoder_norm_types (`Union[str, Tuple[str]]`, defaults to `"rms_norm"`):
            The normalization type(s) to use in the decoder.
        decoder_act_fns (`Union[str, Tuple[str]]`, defaults to `"silu"`):
            The activation function(s) to use in the decoder.
        scaling_factor (`float`, defaults to `1.0`):
            The multiplicative inverse of the root mean square of the latent features. This is used to scale the latent
            space to have unit variance when training the diffusion model. The latents are scaled with the formula `z =
            z * scaling_factor` before being passed to the diffusion model. When decoding, the latents are scaled back
            to the original scale with the formula: `z = 1 / scaling_factor * z`.
    """

    _supports_gradient_checkpointing = False

    def __init__(
        self,
        cfg: DictConfig
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(
            in_channels=self.cfg.in_channels,
            latent_channels=self.cfg.latent_channels,
            attention_head_dim=self.cfg.attention_head_dim,
            block_type=self.cfg.encoder_block_types,
            block_out_channels=self.cfg.encoder_block_out_channels,
            layers_per_block=self.cfg.encoder_layers_per_block,
            qkv_multiscales=self.cfg.encoder_qkv_multiscales,
            downsample_block_type=self.cfg.downsample_block_type,
        )

        self.decoder = Decoder(
            in_channels=self.cfg.in_channels,
            latent_channels=self.cfg.latent_channels,
            attention_head_dim=self.cfg.attention_head_dim,
            block_type=self.cfg.decoder_block_types,
            block_out_channels=self.cfg.decoder_block_out_channels,
            layers_per_block=self.cfg.decoder_layers_per_block,
            qkv_multiscales=self.cfg.decoder_qkv_multiscales,
            norm_type=self.cfg.decoder_norm_types,
            act_fn=self.cfg.decoder_act_fns,
            upsample_block_type=self.cfg.upsample_block_type,
        )

        self.spatial_compression_ratio = 2**(len(self.cfg.encoder_block_out_channels) - 1)
        self.temporal_compression_ratio = 1

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 512
        self.tile_sample_min_width = 512

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 448
        self.tile_sample_stride_width = 448

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled AE decoding. When this option is enabled, the AE will split the input tensor into tiles to compute
        decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width

    def disable_tiling(self) -> None:
        r"""
        Disable tiled AE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced AE decoding. When this option is enabled, the AE will split the input tensor in slices to compute
        decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced AE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x, return_dict=False)[0]

        encoded = self.encoder(x)

        return encoded

    def get_last_layer(self):
        return self.decoder.conv_out.conv.weight

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = False
    ) -> Union[EncoderOutput, Tuple[torch.Tensor]]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.vae.EncoderOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.vae.EncoderOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            encoded = torch.cat(encoded_slices)
        else:
            encoded = self._encode(x)

        return encoded

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = z.shape

        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            return self.tiled_decode(z, return_dict=False)[0]

        decoded = self.decoder(z)

        return decoded

    @apply_forward_hook
    def decode(self, z: torch.Tensor) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.use_slicing and z.size(0) > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        return decoded

    def tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> torch.Tensor:
        raise NotImplementedError('`tiled_encode` has not been implemented for AutoencoderDC.')

    def tiled_decode(
            self,
            z: torch.Tensor,
            return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        raise NotImplementedError(
            '`tiled_decode` has not been implemented for AutoencoderDC.')

    def forward(self, sample: torch.Tensor, return_dict: bool = True) -> torch.Tensor:
        encoded = self.encode(sample, return_dict=False)[0]
        decoded = self.decode(encoded, return_dict=False)[0]
        if not return_dict:
            return (decoded, )
        return DecoderOutput(sample=decoded)
    
    @classmethod
    def _from_pretrained_custom(cls, cfg, path: str, **kwargs) -> "MyAutoencoderDC":
        if is_wandb_run_path(path):
            path = wandb_to_local_path(path)
        elif is_hf_path(path):
            path = hf_to_local_path(path)

        if path.endswith(".safetensors"):
            checkpoint = load_file(path)
        else:
            checkpoint = torch.load(path, map_location="cpu")
        
        # cfg = OmegaConf.create(checkpoint["cfg"])
        # checkpoint.pop("cfg")

        model = cls(cfg)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        return model
    
    @classmethod
    def _from_pretrained_diffuser(clas, path: str, **kwargs) -> "MyAutoencoderDC":
        vae = MyAutoencoderDC.from_pretrained(path, **kwargs)
    
    @classmethod
    def from_pretrained(cls, cfg, **kwargs) -> "MyAutoencoderDC":
        path = cfg.pretrained_path
        if path.startswith("diffuser:"):
            path = path.replace("diffuser:", "")
            return cls._from_pretrained_diffuser(path, **kwargs)
        else:
            return cls._from_pretrained_custom(cfg, path, **kwargs)


class DCAEPreprocessor(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.pretrained_path = cfg.pretrained_path
        self.pretrained_kwargs = cfg.pretrained_kwargs
        self.use_fp16 = cfg.precision == "16-true"
        self.max_encode_length = cfg.max_encode_length
        self.max_decode_length = cfg.logging.max_video_length
        self.log_every_n_batch = cfg.logging.every_n_batch
        super().__init__(cfg)

    def _build_model(self):
        self.vae = MyAutoencoderDC.from_pretrained(
            cfg=self.cfg,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            **self.pretrained_kwargs,
        )

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        raise NotImplementedError(
            "Training not implemented for VAEVideo. Only used for validation"
        )

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        raise NotImplementedError(
            "Testing not implemented for VAEVideo. Only used for validation"
        )
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        #videos, latent_paths = batch
        videos = batch["videos"]
        latent_paths = batch["latent_paths"]
        video_lengths = batch["video_lengths"]
        latent_paths = [Path(path) for path in latent_paths]
        all_done = True
        for latent_path in latent_paths:
            if not latent_path.exists():
                all_done = False
                break
        if all_done:
            return None

        batch_size = videos.shape[0]
        videos = self._rearrange_and_normalize(videos)

        # Encode the video data into a latent space
        # always convert to float16 (as they will be saved as float16 tensors)
        latents = self._encode_videos(videos)
        # latents = latent_dist.sample().to(torch.float16)

        # just to see the progress in wandb
        if batch_idx % 1000 == 0:
            self.log("dummy", 0.0)

        # log gt vs reconstructed video to wandb
        if batch_idx % self.log_every_n_batch == 0 and self.logger:
            reconstructed_videos = self.vae.decode(latents)
            reconstructed_videos = reconstructed_videos.detach().cpu()
            videos = self._rearrange_and_unnormalize(videos, batch_size)
            reconstructed_videos = self._rearrange_and_unnormalize(
                reconstructed_videos, batch_size
            )

            videos = videos.detach().cpu()[:, : self.max_decode_length]
            reconstructed_videos = reconstructed_videos[:, : self.max_decode_length]
            log_video(
                reconstructed_videos,
                videos,
                step=self.global_step,
                namespace="reconstruction_vis",
                logger=self.logger.experiment,
                captions=[
                    f"{p.parent.parent.name}/{p.parent.name}/{p.stem}"
                    for p in latent_paths
                ],
            )
        
        # save the latent to disk
        latents_to_save = (
            rearrange(
                latents,
                "(b f) c h w -> b f c h w",
                b=batch_size,
            )
            .detach()
            .cpu()
        )
        for i, (latent, latent_path) in enumerate(zip(latents_to_save, latent_paths)):
            # should clone latent to avoid having large file size
            safe_torch_save(latent[:video_lengths[i].cpu().item()].clone(), latent_path)
        return None

    def _encode_videos(self, video: torch.Tensor):
        chunks = video.chunk(
            (len(video) + self.max_encode_length - 1) // self.max_encode_length, dim=0
        )
        latent_dist_list = []
        for chunk in chunks:
            latent_dist_list.append(self.vae.encode(chunk))
        return torch.cat(latent_dist_list, dim=0)
    
    def _rearrange_and_normalize(self, videos: torch.Tensor) -> torch.Tensor:
        videos = rearrange(videos, "b f c h w -> (b f) c h w")
        videos = 2.0 * videos - 1.0
        return videos

    def _rearrange_and_unnormalize(
        self, videos: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        videos = 0.5 * videos + 0.5
        videos = rearrange(videos, "(b f) c h w -> b f c h w", b=batch_size)
        return videos
