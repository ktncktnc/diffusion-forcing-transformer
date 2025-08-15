from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from ..base_backbone import BaseBackbone
from .dit_base import DiTBase


class DiT3D(BaseBackbone):

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_type: str,
        external_cond_num_classes: int, # only for label
        external_cond_dim: int,
        use_causal_mask=True,
    ):
        if use_causal_mask:
            raise NotImplementedError(
                "Causal masking is not yet implemented for DiT3D backbone"
            )

        super().__init__(
            cfg,
            x_shape,
            max_tokens,
            external_cond_type=external_cond_type,
            external_cond_num_classes=external_cond_num_classes,
            external_cond_dim=external_cond_dim,
            use_causal_mask=use_causal_mask,
        )

        self.patch_size = cfg.patch_size
        channels, resolution_h, resolution_w, *_ = x_shape

        # assert (
        #     resolution % self.patch_size == 0
        # ), "Resolution must be divisible by patch size."
        self.num_patches_h = resolution_h // self.patch_size
        self.num_patches_w = resolution_w // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        out_channels = self.patch_size**2 * channels

        self.patch_embedder = PatchEmbed(
            img_size=(resolution_h, resolution_w),
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size,
            bias=True,
        )

        self.dit_base = DiTBase(
            num_patches=self.num_patches,
            spatial_grid_size=(self.num_patches_h, self.num_patches_w),
            max_temporal_length=max_tokens,
            out_channels=out_channels,
            variant=cfg.variant,
            pos_emb_type=cfg.pos_emb_type,
            hidden_size=self.hidden_size,
            depth=cfg.depth,
            num_heads=cfg.get("num_heads", None),
            mlp_ratio=cfg.mlp_ratio,
            learn_sigma=False,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
            spatial_mlp_ratio=cfg.get("spatial_mlp_ratio", None),
            embed_col_dim=cfg.get("embed_col_dim", None),
            embed_row_dim=cfg.get("embed_row_dim", None),
            num_col_heads=cfg.get("num_col_heads", None),
            num_row_heads=cfg.get("num_row_heads", None),
            matrix_block=cfg.get("matrix_block", None),
            flatten_matrix_rope=cfg.get("flatten_matrix_rope", None),
            matrix_multi_token=cfg.get("matrix_multi_token", None),
            use_bias=cfg.get("use_bias", None),
            fixed_u=cfg.get('fixed_u', None),
            use_temporal_rope=cfg.get('use_temporal_rope', None)
        )
        self.initialize_weights()

    @property
    def in_channels(self) -> int:
        return self.x_shape[0]

    @staticmethod
    def _patch_embedder_init(embedder: PatchEmbed) -> None:
        # Initialize patch_embedder like nn.Linear (instead of nn.Conv2d):
        w = embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(embedder.proj.bias)

    def initialize_weights(self) -> None:
        self._patch_embedder_init(self.patch_embedder)

        # Initialize noise level embedding and external condition embedding MLPs:
        def _mlp_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.noise_level_pos_embedding.apply(_mlp_init)
        if self.external_cond_embedding is not None:
            self.external_cond_embedding.apply(_mlp_init)

    @property
    def is_matrix_attention(self) -> bool:
        return self.cfg.variant in ["full_matrix_attention", "factorized_matrix_attention"]
    @property
    def noise_level_dim(self) -> int:
        return 256
    
    @property
    def hidden_size(self) -> int:
        if self.is_matrix_attention:
            return self.cfg.embed_row_dim
        else:
            return self.cfg.hidden_size

    @property
    def noise_level_emb_dim(self) -> int:
        if self.is_matrix_attention:
            return self.cfg.embed_row_dim
        else:
            return self.cfg.hidden_size

    @property
    def external_cond_emb_dim(self) -> int:
        if self.external_cond_dim:
            return self.noise_level_emb_dim
        else:
            return 0

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: patchified tensor of shape (B, num_patches, patch_size**2 * C)
        Returns:
            unpatchified tensor of shape (B, H, W, C)
        """
        return rearrange(
            x,
            "b (h w) (p q c) -> b (h p) (w q) c",
            h=self.num_patches_h,
            w=self.num_patches_w,
            p=self.patch_size,
            q=self.patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_batch_size = x.shape[0]
        x = rearrange(x, "b t c h w -> (b t) c h w")

        x = self.patch_embedder(x)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)

        height, width = self.patch_embedder.grid_size

        emb = self.noise_level_pos_embedding(noise_levels)

        if external_cond is not None:
            if self.external_cond_type == 'label':
                cond_emb = self.external_cond_embedding(external_cond.long())
                emb = emb + cond_emb
            elif self.external_cond_type == 'action':
                emb = emb + self.external_cond_embedding(external_cond, external_cond_mask)
            else:
                raise ValueError(
                    f"Unknown external condition type: {self.external_cond_type}. "
                    "Supported types are 'label' and 'action'."
                )
            
        emb = repeat(emb, "b t c -> b (t p) c", p=self.num_patches)

        x = self.dit_base(x, emb, noise_levels, height, width)  # (B, N, C)
        
        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )  # (B * T, H, W, C)
        x = rearrange(
            x, "(b t) h w c -> b t c h w", b=input_batch_size
        )  # (B, T, C, H, W)
        return x
