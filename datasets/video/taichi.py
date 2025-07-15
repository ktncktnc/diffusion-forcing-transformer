from typing import Any, Dict, Optional
import io
import tarfile
import torch
import numpy as np
from PIL import Image
import cv2
import os
from omegaconf import DictConfig
from tqdm import tqdm
from torchvision.datasets.video_utils import _VideoTimestampsDataset, _collate_fn
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT
)
from typing import List, Tuple


class TaichiBaseVideoDataset(BaseVideoDataset):
    _ALL_SPLITS = ["train", "test"]

    def download_dataset(self):
        pass

    def build_metadata(self, split: SPLIT) -> None:
        """
        Build metadata for the dataset and save it in metadata_dir
        This may vary depending on the dataset.
        Default:
        ```
        {
            "video_paths": List[str],
        }
        ```
        """
        video_paths = sorted(list((self.save_dir / split).glob("*/")), key=str)
        video_paths = [str(path) for path in video_paths]
        video_lengths: List[int] = []

        for video_path in tqdm(video_paths, desc=f"Loading {split} metadata"):
            video = os.listdir(video_path)
            video_lengths.append(len(video))

        metadata = {
            "video_paths": video_paths,
            "video_lengths": video_lengths
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def build_transform(self):
        return transforms.Resize(
            (self.resolution, self.resolution),
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=True,
        )
    
    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        """
        Return the length of the video at idx
        """
        return video_metadata["video_lengths"]
    
    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load video from video_idx with given start_frame and end_frame (exclusive)
        if end_frame is None, load until the end of the video
        return shape: (T, C, H, W)
        """
        if end_frame is None:
            end_frame = self.video_length(video_metadata)

        video_path = video_metadata["video_paths"]
        # list dir and sort by frame number
        frame_files = sorted(
            os.listdir(video_path),
            key=lambda x: int(x.split('.')[0])  # assuming filenames are like '0001.png'
        )
        video = []
        for frame_file in frame_files[start_frame:end_frame]:
            frame_path = os.path.join(video_path, frame_file)
            frame = np.array(cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB))
            video.append(frame)
        video = np.stack(video, axis=0)
        return torch.from_numpy(video).permute(0, 3, 1, 2) / 255.0
    
    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        raise NotImplementedError("Taichi only supports unconditional models")
    
    def load_video_and_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load video and condition from video_idx with given start_frame and end_frame (exclusive)
        if end_frame is None, load until the end of the video
        return shape: (T, C, H, W), (T, C)
        """
        raise NotImplementedError("Taichi only supports unconditional models")


class TaichiSimpleVideoDataset(TaichiBaseVideoDataset, BaseSimpleVideoDataset):
    """
    Taichi simple video dataset
    """
    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == 'training':
            split = 'train'
        elif split == "validation":
            split = "test"
        BaseSimpleVideoDataset.__init__(self, cfg, split)


class TaichiAdvancedVideoDataset(
    TaichiBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    Taichi advanced video dataset
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == 'training':
            split = 'train'
        if split == "validation":
            split = "test"
        
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)
