# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding
from timm.models.vision_transformer import Mlp
from typing import Optional, Final
from timm.layers import use_fused_attn
from torch.nn import functional as F


class Attention(nn.Module):
    # Based on: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    # Add attn mask
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            attn_mask: Optional[torch.Tensor] = None,
            rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.rotary_emb = rotary_emb

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        mask = mask if mask is not None else self.attn_mask
        if mask is not None:
            mask = mask.to(x.device)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn + mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x, shift, scale, dim):
    if dim == 1:
        return x * (1 + scale.unsqueeze(dim)) + shift.unsqueeze(dim)
    elif dim == 2:
        T = shift.shape[1]
        x = rearrange(x, 'b (t l) c -> b t l c', t=T)
        x = x * (1 + scale.unsqueeze(dim)) + shift.unsqueeze(dim)
        x = rearrange(x, 'b t l c -> b (t l) c')
        return x
    else:
        raise NotImplementedError

def gate(x, gate, dim):
    if dim == 1:
        return x * gate.unsqueeze(dim)
    elif dim == 2:
        T = gate.shape[1]
        x = rearrange(x, 'b (t l) c -> b t l c', t=T)
        x = x * gate.unsqueeze(dim)
        x = rearrange(x, 'b t l c -> b (t l) c')
        return x
    else:
        raise NotImplementedError

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, :, None].float() * freqs[None, None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, merge_mode=None, **block_kwargs):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if torch.__version__ >= "1.11.0":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
        else:
            def gelu_new(x):
                return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3.0))))
            approx_gelu = lambda: gelu_new
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.merge_mode = merge_mode
        assert self.merge_mode is not None
        if self.merge_mode == "share_norm" or self.merge_mode == "reproduce" or self.merge_mode == "sharenorm_shift_first_t":
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        elif self.merge_mode == "separate_norm":
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm1c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            raise NotImplementedError

    def forward(self, x, t, c_token_len):

        if self.merge_mode == "separate_norm":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=2)
            
            # attention
            cond_tokens = self.norm1(x[:, :c_token_len, :])
            vid_tokens = self.norm1c(x[:, c_token_len:, :])
            vid_tokens = modulate(vid_tokens, shift_msa, scale_msa, dim=2) # (B, T, L, D)
            attn_x = self.attn(torch.cat([cond_tokens, vid_tokens], dim=1)) 
            attn_x = torch.cat([attn_x[:, :c_token_len, :], 
                                gate(attn_x[:, c_token_len:, :], gate_msa, dim=2)], 
                                dim=1)
            x = x + attn_x
            
            # mlp
            cond_tokens = self.norm2(x[:, :c_token_len, :])
            vid_tokens = self.norm2c(x[:, c_token_len:, :])
            vid_tokens = modulate(vid_tokens, shift_mlp, scale_mlp, dim=2) # (B, T, L, D)
            mlp_x = self.mlp(torch.cat([cond_tokens, vid_tokens], dim=1))
            mlp_x = torch.cat([mlp_x[:, :c_token_len, :],
                               gate(mlp_x[:, c_token_len:, :], gate_mlp, dim=2)],
                               dim=1)
            x = x + mlp_x

        elif self.merge_mode == "share_norm":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=2)
            
            # attention
            x = self.norm1(x)
            cond_tokens = x[:, :c_token_len, :]
            vid_tokens = x[:, c_token_len:, :]
            vid_tokens = modulate(vid_tokens, shift_msa, scale_msa, dim=2) # (B, T, L, D)
            attn_x = self.attn(torch.cat([cond_tokens, vid_tokens], dim=1)) 
            attn_x = torch.cat([attn_x[:, :c_token_len, :], 
                                gate(attn_x[:, c_token_len:, :], gate_msa, dim=2)], 
                                dim=1)
            x = x + attn_x
            
            # mlp
            x = self.norm2(x)
            cond_tokens = x[:, :c_token_len, :]
            vid_tokens = x[:, c_token_len:, :]
            vid_tokens = modulate(vid_tokens, shift_mlp, scale_mlp, dim=2) # (B, T, L, D)
            mlp_x = self.mlp(torch.cat([cond_tokens, vid_tokens], dim=1))
            mlp_x = torch.cat([mlp_x[:, :c_token_len, :],
                               gate(mlp_x[:, c_token_len:, :], gate_mlp, dim=2)],
                               dim=1)
            x = x + mlp_x
        
        elif self.merge_mode == "reproduce":
            # attention
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t[:, 0]).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa, dim=1))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp, dim=1))
        
        elif self.merge_mode == "sharenorm_shift_first_t":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=2)
            
            # attention
            x = self.norm1(x)
            cond_tokens = modulate(x[:, :c_token_len, :], shift_msa[:, 0], scale_msa[:, 0], dim=1)
            vid_tokens = modulate(x[:, c_token_len:, :], shift_msa, scale_msa, dim=2) # (B, T, L, D)
            attn_x = self.attn(torch.cat([cond_tokens, vid_tokens], dim=1)) 
            attn_x = torch.cat([attn_x[:, :c_token_len, :], 
                                gate(attn_x[:, c_token_len:, :], gate_msa, dim=2)], 
                                dim=1)
            x = x + attn_x
            
            # mlp
            x = self.norm2(x)
            cond_tokens = modulate(x[:, :c_token_len, :], shift_msa[:, 0], scale_msa[:, 0], dim=1)
            vid_tokens = modulate(x[:, c_token_len:, :], shift_msa, scale_msa, dim=2) # (B, T, L, D)
            mlp_x = self.mlp(torch.cat([cond_tokens, vid_tokens], dim=1))
            mlp_x = torch.cat([mlp_x[:, :c_token_len, :],
                               gate(mlp_x[:, c_token_len:, :], gate_mlp, dim=2)],
                               dim=1)
            x = x + mlp_x
        
        else:
            raise NotImplementedError
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DIT1D(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            cfg,
            x_shape,
            max_tokens,
            external_cond_type,
            external_cond_num_classes,
            external_cond_dim,
            use_causal_mask
            # num_frames=16,
            # n_token_per_frame=32,
            # in_channels=12,
            # hidden_size=1152,
            # depth=28,
            # num_heads=16,
            # mlp_ratio=4.0,
            # learn_sigma=True,
            # use_fp16=False,
            # context_dim=768,
            # context_token=77,
            # merge_mode=None,
            # causal_attn_mode=None,
            # use_rotary_emb=False,
            # qk_norm=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.x_shape = x_shape

        self.learn_sigma = cfg.learn_sigma
        self.in_channels = x_shape[0]
        print('self.in_channels:', self.in_channels)
        print('self.in_channels:', self.in_channels)
        print('self.in_channels:', self.in_channels)
        print('self.in_channels:', self.in_channels)
        self.out_channels = self.in_channels * 2 if cfg.learn_sigma else self.in_channels
        self.n_token_per_frame = x_shape[2]
        print('self.n_token_per_frame:', self.n_token_per_frame)
        print('self.n_token_per_frame:', self.n_token_per_frame)
        print('self.n_token_per_frame:', self.n_token_per_frame)
        print('self.n_token_per_frame:', self.n_token_per_frame)
        self.num_frames = max_tokens
        self.num_heads = cfg.num_heads
        self.use_rotary_emb = cfg.use_rotary_emb

        context_token = 0

        self.causal_attn_mode = cfg.causal_attn_mode
        if self.causal_attn_mode is None:
            attn_mask = None
        
        elif self.causal_attn_mode == "temporal_causal":
            """ attn_mask
            tensor([[0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
                    [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
                    [0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
                    [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
                    [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
                    [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
                    [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
            """
            video_attn_mask = torch.zeros((self.num_frames, self.num_frames))
            video_attn_mask = video_attn_mask.fill_(float('-inf')).triu_(1)
            video_attn_mask = repeat(video_attn_mask, 't1 t2 -> (t1 n1) (t2 n2)', 
                                     n1=self.n_token_per_frame, n2=self.n_token_per_frame)
            total_token_len = context_token + self.num_frames * self.n_token_per_frame
            attn_mask = torch.zeros((total_token_len, total_token_len))
            attn_mask[:context_token, context_token:] = -torch.inf
            attn_mask[-self.num_frames * self.n_token_per_frame:, -self.num_frames * self.n_token_per_frame:] = video_attn_mask
        
        elif self.causal_attn_mode == "video_temporal_causal":
            """ attn_mask                                                                                
            tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],            
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],        
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],        
                    [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
                    [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
                    [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
                    [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
            """
            video_attn_mask = torch.zeros((self.num_frames, self.num_frames))
            video_attn_mask = video_attn_mask.fill_(float('-inf')).triu_(1)
            video_attn_mask = repeat(video_attn_mask, 't1 t2 -> (t1 n1) (t2 n2)', 
                                     n1=self.n_token_per_frame, n2=self.n_token_per_frame)
            total_token_len = context_token + self.num_frames * self.n_token_per_frame
            attn_mask = torch.zeros((total_token_len, total_token_len))
            attn_mask[-self.num_frames * self.n_token_per_frame:, -self.num_frames * self.n_token_per_frame:] = video_attn_mask
        
        else:
            raise NotImplementedError

        self.x_embedder = nn.Linear(self.in_channels, self.cfg.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(self.cfg.hidden_size)
        if external_cond_dim:
            self.c_embedder = nn.Linear(external_cond_dim, self.external_cond_emb_dim)

        # Will use fixed sin-cos embedding:
        self.num_tokens = self.n_token_per_frame * self.num_frames
        # Will use fixed sin-cos embedding:
        if self.cfg.use_rotary_emb:
            self.rotary_emb = RotaryEmbedding(dim=self.cfg.hidden_size // self.cfg.num_heads)
        else:
            self.rotary_emb = None
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.cfg.hidden_size), requires_grad=False)
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(self.cfg.hidden_size, self.cfg.num_heads, mlp_ratio=self.cfg.mlp_ratio, 
                     merge_mode=self.cfg.merge_mode, attn_mask=attn_mask, 
                     qk_norm=self.cfg.qk_norm, rotary_emb=self.rotary_emb) 
                     for _ in range(self.cfg.depth)
        ])
        self.final_layer = nn.Sequential(
            nn.LayerNorm(self.cfg.hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(self.cfg.hidden_size, self.out_channels)
        )
        self.initialize_weights()

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size if self.external_cond_dim else 0
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if not self.use_rotary_emb:
            pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_tokens)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer[1].weight, 0)
        nn.init.constant_(self.final_layer[1].bias, 0)

    def forward(self, x, t, context=None, external_cond_mask=None):
        """
        Forward pass of DiT.
        x: (N, T, L, C) tensor of spatial inputs (images or latent representations of images)
        t: (N, T, ) tensor of diffusion timesteps
        c: (N, L, C) tensor of class labels
        """
        B, T = x.shape[:2]
        if len(x.shape) == 5:
            squeeze = True
            x = x[..., 0, :]
        else:
            squeeze = False

        x = rearrange(x, 'b t c l -> b (t l) c')
        if self.use_rotary_emb:
            x = self.x_embedder(x)
        else:
            x = self.x_embedder(x) + self.pos_embed  # (N, T * L, D), where T = H * W / patch_size ** 2
        
        if context is not None:
            c = self.c_embedder(context) # (N, L, D), txt representation
            x = torch.cat([c, x], dim=1)
            context_len = c.shape[1]
        else:
            context_len = 0

        t = self.t_embedder(t)  # (N, T, D)
        for block in self.blocks:
            x = block(x, t, context_len)  # (N, T, D)
        x = self.final_layer(x)  # (N, T, patch_size ** 2 * out_channels)
        x = x[:, context_len:, :]
        x = rearrange(x, 'b (t l) c -> b t c l', t=T)

        if squeeze:
            x = x.unsqueeze(-2)
        return x

    # def forward_with_cfg(self, x, t, y, cfg_scale):
    #     """
    #     Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
    #     """
    #     # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    #     half = x[: len(x) // 2]
    #     combined = torch.cat([half, half], dim=0)
    #     model_out = self.forward(combined, t, y)
    #     # For exact reproducibility reasons, we apply classifier-free guidance on only
    #     # three channels by default. The standard approach to cfg applies it to all channels.
    #     # This can be done by uncommenting the following line and commenting-out the line following that.
    #     # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
    #     eps, rest = model_out[:, :3], model_out[:, 3:]
    #     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    #     half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    #     eps = torch.cat([half_eps, half_eps], dim=0)
    #     return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_pos_embed(embed_dim, token_len, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    pos = np.arange(token_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed(embed_dim, n_frame, n_token_per_frame, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_nframe = np.arange(n_frame, dtype=np.float32)
    grid_ntoken = np.arange(n_token_per_frame, dtype=np.float32)
    grid = np.meshgrid(grid_nframe, grid_ntoken)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, n_frame, n_token_per_frame])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
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
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# #################################################################################
# #                                   DiT Configs                                  #
# #################################################################################

# def DiT_XL_2(**kwargs):
#     return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

# def DiT_XL_4(**kwargs):
#     return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

# def DiT_XL_8(**kwargs):
#     return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

# def DiT_L_2(**kwargs):
#     return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

# def DiT_L_4(**kwargs):
#     return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

# def DiT_L_8(**kwargs):
#     return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

# def DiT_B_2(**kwargs):
#     return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

# def DiT_B_4(**kwargs):
#     return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

# def DiT_B_8(**kwargs):
#     return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

# def DiT_S_2(**kwargs):
#     return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

# def DiT_S_4(**kwargs):
#     return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

# def DiT_S_8(**kwargs):
#     return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


# DiT_models = {
#     'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
#     'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
#     'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
#     'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
# }