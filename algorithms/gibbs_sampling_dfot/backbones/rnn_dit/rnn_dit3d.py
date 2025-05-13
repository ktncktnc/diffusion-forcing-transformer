from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from ..base_backbone import BaseBackbone
from .dit_base import DiTBase
from .clstm import ConvLSTM


class RNN_DiT3D(BaseBackbone):

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_frames: int,
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
            max_frames,
            max_tokens,
            external_cond_type,
            external_cond_num_classes,
            external_cond_dim,
            use_causal_mask,
        )

        hidden_size = cfg.hidden_size
        self.patch_size = cfg.patch_size
        channels, resolution, *_ = x_shape
        assert (
            resolution % self.patch_size == 0
        ), "Resolution must be divisible by patch size."
        self.num_patches = (resolution // self.patch_size) ** 2
        out_channels = self.patch_size**2 * channels

        self.patch_embedder = PatchEmbed(
            img_size=resolution,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=hidden_size,
            bias=True,
        )

        self.dit_base = DiTBase(
            num_patches=self.num_patches,
            max_temporal_length=max_tokens,
            out_channels=out_channels,
            variant=cfg.variant,
            pos_emb_type=cfg.pos_emb_type,
            hidden_size=hidden_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            learn_sigma=False,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        )

        self.rnn = ConvLSTM(
            input_dim=self.in_channels,
            hidden_dim=cfg.conv_lstm.hidden_dim,
            kernel_size=cfg.conv_lstm.kernel_size,
            num_layers=cfg.conv_lstm.num_layers,
            batch_first=True,
            bias=True
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
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.cfg.hidden_size

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size if self.external_cond_dim else 0
    
    @property
    def frame_idx_emd_dim(self) -> int:
        return self.cfg.hidden_size 

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
            h=int(self.num_patches**0.5),
            p=self.patch_size,
            q=self.patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
        gibbs_frame_idx: Optional[int] = None,
    ) -> torch.Tensor:
        print('self.cfg', self.cfg)
        input_batch_size, n_frames = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.patch_embedder(x)
        x = rearrange(x, "(b t) p c -> b (t p) c", b=input_batch_size)
        if self.cfg.gibbs.enabled:
            x, mask = self.gibbs_preprocessing(
                tokens=x,
                n_frames=n_frames,
                n_tokens_per_frame=self.num_patches,
                frame_idx=gibbs_frame_idx,
                device=x.device,
            )
        else:
            mask = None

        emb = self.noise_level_pos_embedding(noise_levels)

        if gibbs_frame_idx is not None:
            # Add frame index embedding
            if gibbs_frame_idx.dim() == 1:
                gibbs_frame_idx = gibbs_frame_idx.unsqueeze(1)
            frame_idx_emb = self.frame_idx_embedding(gibbs_frame_idx)
            emb = emb + frame_idx_emb
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

        x = self.dit_base(x, emb, attn_mask=mask)  # (B, N, C)
        
        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )  # (B * T, H, W, C)
        x = rearrange(
            x, "(b t) h w c -> b t c h w", b=input_batch_size
        )  # (B, T, C, H, W)

        # RNN
        x, _ = self.rnn(x)

        return x

    def gibbs_preprocessing(self, tokens: torch.Tensor, n_frames: int, n_tokens_per_frame: int, frame_idx, device) -> torch.Tensor:
        """
        Generate a Gibbs attention mask for the given number of frames and tokens per frame.
        Args:
            n_frames: Number of frames.
            n_tokens_per_frame: Number of tokens per frame. In case of factorized attention, n_tokens_per_frame = 1
        Returns:
            A tensor representing the attention mask.
        """
        if not self.cfg.gibbs.enabled:
            raise ValueError("Gibbs sampling is not enabled in the config.")

        # assert self.cfg.gibbs.enabled and frame_idx is not None, "Gibbs sampling is enabled but frame_idx is None."
        
        match self.cfg.gibbs.mask_type:
            # Do not do anything
            case None:
                mask = None
            # Mask out the tokens in the frame_idx
            case "bert":
                tokens = rearrange(tokens, "b (t p) c -> b t p c", t=n_frames)
                tokens[torch.arange(tokens.shape[0]), frame_idx] = 0
                tokens = rearrange(tokens, "b t p c -> b (t p) c")
                mask = None
            # Set mask to 0 for the diagonal values of the frame_idx
            case "diagonal_gibbs":
                mask = torch.ones((frame_idx.shape[0], n_frames*n_tokens_per_frame, n_frames*n_tokens_per_frame), device=device)
                for i in range(len(frame_idx)):
                    mask[i, frame_idx[i]*n_tokens_per_frame:(frame_idx[i]+1)*n_tokens_per_frame, frame_idx[i]*n_tokens_per_frame:(frame_idx[i]+1)*n_tokens_per_frame] = 0
            # Set mask to 0 for the whole row of the frame_idx (TODO: check)
            case "gibbs":
                mask = torch.ones((frame_idx.shape[0], n_frames*n_tokens_per_frame, n_frames*n_tokens_per_frame), device=device)
                for i in range(len(frame_idx)):
                    mask[i, :, frame_idx[i]*n_tokens_per_frame:(frame_idx[i]+1)*n_tokens_per_frame] = 0
            # Set mask = 0 for the diagonal values of the matrix
            case "diagonal":
                mask = torch.ones((frame_idx.shape[0], n_frames*n_tokens_per_frame, n_frames*n_tokens_per_frame), device=device)
                for i in range(len(frame_idx)):
                    mask[:, frame_idx[i]*n_tokens_per_frame:(frame_idx[i]+1)*n_tokens_per_frame, frame_idx[i]*n_tokens_per_frame:(frame_idx[i]+1)*n_tokens_per_frame] = 0
            case "token_hollow":
                mask = 1 - torch.eye(n_frames*n_tokens_per_frame, device=device)
                mask = mask.unsqueeze(0).repeat(tokens.shape[0], 1, 1)
            case "frame_hollow":
                mask = torch.ones((tokens.shape[0], n_frames*n_tokens_per_frame, n_frames*n_tokens_per_frame), device=device)
                for i in range(n_frames):
                    mask[:, i* n_tokens_per_frame:(i+1)*n_tokens_per_frame, i*n_tokens_per_frame:(i+1)*n_tokens_per_frame] = 0
            case _:
                raise ValueError(f"Unknown mask type: {self.cfg.gibbs.mask_type}. Supported types are None, 'bert', 'diagonal_gibbs', 'gibbs', and 'diagonal'.")
        
        return tokens, mask
