from abc import abstractmethod, ABC
from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig
from .modules.embeddings import (
    StochasticTimeEmbedding,
    RandomDropoutCondEmbedding,
)
from diffusers.models.embeddings import LabelEmbedding


class BaseBackbone(ABC, nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        external_cond_type: str = 'action',
        external_cond_num_classes: int = None,  # only for label
        use_causal_mask=True,
    ):

        super().__init__()

        self.cfg = cfg
        self.external_cond_type = external_cond_type
        self.external_cond_num_classes = external_cond_num_classes
        self.external_cond_dim = external_cond_dim
        self.use_causal_mask = use_causal_mask
        self.x_shape = x_shape

        self.noise_level_pos_embedding = StochasticTimeEmbedding(
            dim=self.noise_level_dim,
            time_embed_dim=self.noise_level_emb_dim,
            use_fourier=self.cfg.get("use_fourier_noise_embedding", False),
        )
        self.external_cond_embedding = self._build_external_cond_embedding()

    def _build_external_cond_embedding(self) -> Optional[nn.Module]:
        if not self.external_cond_dim:
            return None
        
        if self.external_cond_type == 'label':
            return LabelEmbedding(
                self.external_cond_num_classes,
                self.external_cond_emb_dim,
                dropout_prob=self.cfg.get("external_cond_dropout", 0.1),
            )
        elif self.external_cond_type == 'action':
            return RandomDropoutCondEmbedding(
                    self.external_cond_dim,
                    self.external_cond_emb_dim,
                    dropout_prob=self.cfg.get("external_cond_dropout", 0.0),
            )
        else:
            raise ValueError(
                f"Unknown external condition type: {self.external_cond_type}. "
                "Supported types are 'label' and 'action'."
            )    

    @property
    def noise_level_dim(self):
        return max(self.noise_level_emb_dim // 4, 32)

    @property
    @abstractmethod
    def noise_level_emb_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def external_cond_emb_dim(self):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
        external_cond_mask: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError
