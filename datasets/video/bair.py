"""
Adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/kinetics.py
"""

from typing import Any, Dict, List, Optional, Literal
from fractions import Fraction
import csv
from tqdm.contrib.concurrent import process_map
import os
import random
from os import path
from pathlib import Path
import urllib
import shutil
from multiprocessing import Pool
from functools import partial
from omegaconf import DictConfig
import torch
from torchvision.io import write_video
from torchvision.datasets.utils import (
    download_and_extract_archive,
    check_integrity,
    download_url,
)
from torchvision.datasets.video_utils import _VideoTimestampsDataset, _collate_fn
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import numpy as np
from utils.print_utils import cyan
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT,
)
from .utils import read_video, rescale_and_crop
from utils.augmentation import AugmentPipe


class BAIRBaseVideoDataset(BaseVideoDataset):
    @property
    def use_video_preprocessing(self) -> bool:
        return self.cfg.video_preprocessing is not None

    def _should_download(self) -> bool:
        return False
    
    def download_dataset(self) -> None:        
        pass

    def build_metadata(self, split: SPLIT) -> None:
        """
        Build metadata for the dataset and save it in metadata_dir
        This may vary depending on the dataset.
        Default:
        ```
        {
            "video_paths": List[str],
            "video_pts": List[str],
            "video_fps": List[float],
        }
        ```
        """       
        if split == "training":
            split_name = 'train'
        else:
            split_name = 'test'
        
        video_paths = sorted(list((self.save_dir / "softmotion30_44k" / split_name / "video_aux1").glob("**/*.mp4")), key=str)
        dl: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(video_paths),
            batch_size=16,
            num_workers=64,
            collate_fn=_collate_fn,
        )
        video_pts: List[torch.Tensor] = (
            []
        )  # each entry is a tensor of shape (num_frames, )
        video_fps: List[float] = []

        with tqdm(total=len(dl), desc=f"Building metadata for {split}") as pbar:
            for batch in dl:
                pbar.update(1)
                batch_pts, batch_fps = list(zip(*batch))
                batch_pts = [
                    torch.as_tensor(pts, dtype=torch.long) for pts in batch_pts
                ]
                video_pts.extend(batch_pts)
                video_fps.extend(batch_fps)

        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def get_latent_paths(self, split: SPLIT) -> List[Path]:
        """
        Return list of latent paths for the given split
        """
        metadata = torch.load(self.metadata_dir / f"{split}.pt", weights_only=False)
        return sorted([self.video_metadata_to_latent_path({key: metadata[key][i] for key in metadata.keys()}) for i in range(len(metadata["video_paths"]))], key=str)
    
class BAIRSimpleVideoDataset(
    BAIRBaseVideoDataset, BaseSimpleVideoDataset
):
    """
    BAIR simple video dataset
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        BaseSimpleVideoDataset.__init__(self, cfg, split)
        # self.setup()


class BAIRAdvancedVideoDataset(
    BAIRBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    BAIR advanced video dataset
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "validation":
            split = "test"
        
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        raise NotImplementedError("BAIR only supports unconditional models")

    