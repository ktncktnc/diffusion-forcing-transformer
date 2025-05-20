"""
Adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/kinetics.py
"""

from typing import Any, Dict, List, Optional, Literal
from fractions import Fraction
import torch
from pathlib import Path
from torchvision.io import write_video
from omegaconf import DictConfig

import torch.nn.functional as F
import numpy as np
from utils.print_utils import cyan
from .ucf_101 import UCF101BaseVideoDataset, UCF101SimpleVideoDataset, UCF101AdvancedVideoDataset, VideoPreprocessingMp4FPS
from .utils import read_video, rescale_and_crop
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT,
)

class SplitUCF101BaseVideoDataset(UCF101BaseVideoDataset):
    @property
    def video_split_percent(self) -> float:
        return self.cfg.video_split_percent
    
    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load metadata from metadata_dir
        """
        splits = ['training', 'validation']
        metadata = []
        for split in splits:
            metadata.append(torch.load(self.metadata_dir / f"{split}.pt", weights_only=False))
        
        final_metadata = {
            "video_paths": metadata[0]['video_paths'] + metadata[1]['video_paths'],
            "labels": metadata[0]['labels'] + metadata[1]['labels'],
            "video_pts": metadata[0]['video_pts'] + metadata[1]['video_pts'],
            "video_fps": metadata[0]['video_fps'] + metadata[1]['video_fps']
        }
        return [
            {key: final_metadata[key][i] for key in final_metadata.keys()}
            for i in range(len(final_metadata["video_paths"]))
        ]

    def get_latent_paths(self, split: SPLIT) -> List[Path]:
        """
        Return list of latent paths for the given split
        """
        training_latents = super().get_latent_paths('training')
        validation_latents = super().get_latent_paths('validation')

        return sorted(training_latents+validation_latents, key=str)

    def video_length(self, video_metadata: Dict[str, Any], split=None) -> int:
        """
        Return the length of the video at idx
        """
        split = split or self.split
        total_length = len(video_metadata["video_pts"])
        if split == "training":
            return round(total_length * self.video_split_percent)
        else:
            return total_length - round(total_length * self.video_split_percent)
    
    def prepare_clips(self) -> None:
        """
        Compute cumulative sizes for the dataset and update self.cumulative_sizes
        Shuffle the dataset with a fixed seed
        """
        num_clips = torch.as_tensor(
            [
                max(self.video_length(video_metadata) - self.n_frames + 1, 1)
                for video_metadata in self.metadata
            ]
        )
        self.cumulative_sizes = num_clips.cumsum(0).tolist()
        self.idx_remap = self._build_idx_remap()

    def get_split_start_end_frame(self, metadata, start_frame: int, end_frame: int) -> tuple:
        """
        Get the start and end frame for the split
        """           
        if self.split == "training":
            if end_frame is None:
                end_frame = self.video_length(metadata)
            return start_frame, end_frame
        
        val_start_frame = self.video_length(metadata, 'training')
        if end_frame is None:
            end_frame = self.video_length(metadata, 'validation')
        return val_start_frame+start_frame, val_start_frame+end_frame

    def load_video(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int=None
    ) -> torch.Tensor:
        
        start_frame, end_frame = self.get_split_start_end_frame(
            video_metadata, start_frame, end_frame
        )

        return super().load_video(
            video_metadata, start_frame, end_frame
        )

    def load_latent(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        start_frame, end_frame = self.get_split_start_end_frame(video_metadata, start_frame, end_frame)
        latent = super().load_latent(
            video_metadata, start_frame, end_frame
        )
        return latent


class SplitUCF101SimpleVideoDataset(
    UCF101SimpleVideoDataset, SplitUCF101BaseVideoDataset
):
    """
    UCF-101 simple video dataset
    """

    pass


class SplitUCF101AdvancedVideoDataset(
    SplitUCF101SimpleVideoDataset, UCF101AdvancedVideoDataset
):
    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        UCF101AdvancedVideoDataset.__init__(self, cfg, split, current_epoch)