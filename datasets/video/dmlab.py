from typing import Any, Dict, Optional
import io
import tarfile
import torch
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT
)
from typing import List, Tuple


class DMLabBaseVideoDataset(BaseVideoDataset):
    _ALL_SPLITS = ["training", "validation"]

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
        video_paths = sorted(list((self.save_dir / split).glob("**/*.npz")), key=str)
        video_paths = [str(path) for path in video_paths]
        video_lengths: List[int] = []
        
        for video_path in tqdm(video_paths, desc=f"Loading {split} metadata"):
            video = np.load(video_path)
            video_lengths.append(video["video"].shape[0])

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
        video = np.load(video_path)["video"][start_frame:end_frame]
        return torch.from_numpy(video).permute(0, 3, 1, 2) / 255.0

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        path = video_metadata["video_paths"]
        actions = np.load(path)["actions"][start_frame:end_frame]
        return torch.from_numpy(np.eye(3)[actions]).float()
    
    def load_video_and_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load video and conditions from video_idx with given start_frame and end_frame (exclusive)
        """
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        
        data = np.load(video_metadata["video_paths"])
        video = data["video"][start_frame:end_frame]
        actions = data["actions"][start_frame:end_frame]
        return torch.from_numpy(video).permute(0, 3, 1, 2) / 255.0, torch.from_numpy(np.eye(3)[actions]).float()


class DMLabSimpleVideoDataset(DMLabBaseVideoDataset, BaseSimpleVideoDataset):
    """
    Minecraft simple video dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == "test":
            split = "validation"
        BaseSimpleVideoDataset.__init__(self, cfg, split)


class DMLabAdvancedVideoDataset(
    DMLabBaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    Minecraft advanced video dataset
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: str = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "test":
            split = "validation"
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        path = video_metadata["video_paths"]
        actions = np.load(path)["actions"][start_frame:end_frame]
        return torch.from_numpy(np.eye(3)[actions]).float()
