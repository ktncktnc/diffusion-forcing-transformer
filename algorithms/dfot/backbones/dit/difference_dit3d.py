from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from ..base_backbone import BaseBackbone
from .dit_base import DiTBase
from diffusers.models.embeddings import LabelEmbedding


class DifferenceDiT3D(BaseBackbone):

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
            max_tokens*2, # doubling max_tokens for difference encoding
            external_cond_type=external_cond_type,
            external_cond_num_classes=external_cond_num_classes,
            external_cond_dim=external_cond_dim,
            use_causal_mask=use_causal_mask,
        )
        self.merge_type = cfg.merge_type
        assert self.merge_type in ["concat", "interleaved"], f"Unsupported merge type: {self.merge_type}"
        hidden_size = cfg.hidden_size
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
            embed_dim=hidden_size,
            bias=True,
        )

        # Embed for x and diff tensors: 0 for x, 1 for diff
        self.diff_embedder = LabelEmbedding(
                2,
                self.cfg.hidden_size,
                dropout_prob=0.0,
            )

        self.dit_base = DiTBase(
            num_patches=self.num_patches,
            spatial_grid_size=(self.num_patches_h, self.num_patches_w),
            max_temporal_length=max_tokens*2,  # doubling max_tokens for difference encoding
            out_channels=out_channels,
            variant=cfg.variant,
            pos_emb_type=cfg.pos_emb_type,
            hidden_size=hidden_size,
            depth=cfg.depth,
            num_heads=cfg.get("num_heads", None),
            mlp_ratio=cfg.mlp_ratio,
            learn_sigma=False,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
            embed_col_dim=cfg.get("embed_col_dim", None),
            embed_row_dim=cfg.get("embed_row_dim", None),
            num_col_heads=cfg.get("num_col_heads", None),
            num_row_heads=cfg.get("num_row_heads", None),
            matrix_block=cfg.get("matrix_block", None),
            flatten_matrix_rope=cfg.get("flatten_matrix_rope", None),
            matrix_multi_token=cfg.get("matrix_multi_token", None),
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
       
        self.diff_embedder.apply(_mlp_init)
        self.noise_level_pos_embedding.apply(_mlp_init)
        if self.external_cond_embedding is not None:
            self.external_cond_embedding.apply(_mlp_init)

    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.cfg.hidden_size

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size if self.external_cond_dim else 0

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

    def create_diff_index(self, x: torch.Tensor, diff_first=True) -> torch.Tensor:
        """
        Create a tensor of shape (B,T) with 0 for x and 1 for diff.
        """
        batch_size, time_steps = x.shape[:2]
        time_steps = time_steps // 2  # Assuming x and diff are interleaved
        x = torch.zeros((batch_size, time_steps), dtype=torch.long, device=x.device)
        diff = torch.ones((batch_size, time_steps), dtype=torch.long, device=x.device)
        x = [diff, x] if diff_first else [x, diff]
        if self.merge_type == "interleaved":
            return rearrange(torch.stack(x, dim=-1), "b t c -> b (t c)")
        elif self.merge_type == "concat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError(f"Unsupported merge type: {self.merge_type}. Supported types are 'concat' and 'interleaved'.")
        
    def make_diff_index_embedding(self, x: torch.Tensor, diff_first=True) -> torch.Tensor:
        idx = self.create_diff_index(x, diff_first=diff_first)
        idx = rearrange(idx, "b t -> (b t)")
        diff_emb = self.diff_embedder(idx)
        diff_emb = rearrange(diff_emb, "(b t) c -> b t c", b=x.shape[0])
        return diff_emb

    def forward(
        self,
        x: torch.Tensor,
        # diff: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_batch_size = x.shape[0]
        length = x.shape[1]
        diff_emb = self.make_diff_index_embedding(x, diff_first=True)
        x = rearrange(x, "b t c h w -> (b t) c h w")

        x = self.patch_embedder(x)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)
        height, width = self.patch_embedder.grid_size
        emb = diff_emb + self.noise_level_pos_embedding(noise_levels)
        if external_cond is not None:
            if self.external_cond_type == 'label':
                cond_emb = self.external_cond_embedding(external_cond.long())
                cond_emb = repeat(cond_emb, "b two d -> b (two p) d", p=length//2)
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
