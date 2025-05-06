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

VideoPreprocessingType = Literal["npz", "mp4"]
VideoPreprocessingMp4FPS: int = 10


# def _dl_wrap(tarpath: str, videopath: str, line: str) -> None:
#     download_and_extract_archive(line, tarpath, videopath, remove_finished=True)


def _preprocess_video(
    video_path: Path,
    resolution: int,
    preprocessing_type: VideoPreprocessingType = "npz",
):
    try:
        preprocessed_video_path = (
            video_path.parent.parent.parent
            / f"preprocessed_{resolution}_{preprocessing_type}"
            / video_path.parent.name
            / video_path.name
        )
        
        if preprocessed_video_path.exists():
            # print(f"Preprocessed video already exists: {preprocessed_video_path}")
            return
        video = read_video(str(video_path))
        video = rescale_and_crop(video, resolution)
        # create directory if it doesn't exist
        video_path.parent.mkdir(parents=True, exist_ok=True)

        if preprocessing_type == "npz":               
            np.savez_compressed(
                preprocessed_video_path,
                video=video.transpose(0, 3, 1, 2).copy(),
            )
        elif preprocessing_type == "mp4":
            # write video
            write_video(
                filename=preprocessed_video_path,
                video_array=torch.from_numpy(video),
                fps=VideoPreprocessingMp4FPS,
            )

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    # remove original video
    # video_path.unlink()


class UCF101BaseVideoDataset(BaseVideoDataset):
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
        with open(os.path.join(self.save_dir, f"{split}03.json"), "r") as f:
            video_list = json.load(f)
            video_paths = [(self.save_dir / v['video_path'].replace('datasets/ucf101/', '')) for v in video_list]
            labels = [v['label'] for v in video_list]

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
            "labels": labels,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def get_latent_paths(self, split: SPLIT) -> List[Path]:
        """
        Return list of latent paths for the given split
        """
        metadata = torch.load(self.metadata_dir / f"{split}.pt", weights_only=False)
        paths = [self.video_metadata_to_latent_path({key: metadata[key][i] for key in metadata.keys()}) for i in range(len(metadata["video_paths"]))]
        # check if paths exist
        paths = [path for path in paths if path.exists()]
        return paths

    def setup(self) -> None:
        if self.use_video_preprocessing:
            # if not (
            #     self.save_dir
            #     / f"preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
            # ).exists():
            for split in ["training", "validation", "test"]:
                print(f'Preprocessing videos for {split}...')
                self._preprocess_videos(split)
            self.metadata = self.exclude_failed_videos(self.metadata)
            self.transform = lambda x: x

    def _preprocess_videos(self, split: SPLIT) -> None:
        """
        Preprocesses videos to {self.resolution}x{self.resolution} resolution
        """
        print(
            cyan(
                f"Preprocessing {split} videos to {self.resolution}x{self.resolution}..."
            )
        )
        (
            self.save_dir
            / f"preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
        ).mkdir(parents=True, exist_ok=True)
        video_paths = torch.load(self.metadata_dir / f"{split}.pt", weights_only=False)
        video_paths = video_paths["video_paths"]
        preprocess_fn = partial(
            _preprocess_video,
            resolution=self.resolution,
            preprocessing_type=self.cfg.video_preprocessing
        )
        with Pool(8) as pool, tqdm(total=len(video_paths), desc=f"Preprocessing {split} videos") as pbar:
            for result in pool.imap(preprocess_fn, video_paths):
                pbar.update()
                pbar.refresh()

        print('Done!')

    def exclude_failed_videos(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Exclude videos that failed to preprocess
        """
        preprocessed_video_paths = set(
            list(
                (
                    self.save_dir
                    / f"preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
                ).glob(f"**/*.{self.cfg.video_preprocessing}")
            )
        )
        return self.subsample(
            metadata,
            lambda video_metadata: self.video_path_to_preprocessed_path(
                video_metadata["video_paths"]
            )
            in preprocessed_video_paths,
            "failed-to-preprocess videos",
        )

    def video_path_to_preprocessed_path(self, video_path: Path) -> Path:
        a = (
            video_path.parent.parent.parent
            / f"preprocessed_{self.resolution}_{self.cfg.video_preprocessing}"
            / video_path.parent.name
            / video_path.name
        )
        return a

    def load_video(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int=None
    ) -> torch.Tensor:
        try:
            if end_frame is None:
                end_frame = self.video_length(video_metadata)
                
            if self.use_video_preprocessing:
                preprocessed_path = self.video_path_to_preprocessed_path(
                    video_metadata["video_paths"]
                )
                match self.cfg.video_preprocessing:
                    case "npz":
                        video = np.load(
                            preprocessed_path,
                        )[
                            "video"
                        ][start_frame:end_frame]
                        return torch.from_numpy(video / 255.0).float()
                    case "mp4":
                        video = read_video(
                            preprocessed_path,
                            pts_unit="sec",
                            start_pts=Fraction(start_frame, VideoPreprocessingMp4FPS),
                            end_pts=Fraction(end_frame - 1, VideoPreprocessingMp4FPS),
                        )
                        return video.permute(0, 3, 1, 2) / 255.0
            else:
                return super().load_video(video_metadata, start_frame, end_frame)
        except Exception as e:
            print(f"Error loading video {video_metadata['video_paths']}: {e}")
            return None

class UCF101SimpleVideoDataset(
    UCF101BaseVideoDataset, BaseSimpleVideoDataset
):
    """
    UCF-101 simple video dataset
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        BaseSimpleVideoDataset.__init__(self, cfg, split)
        self.setup()


class UCF101AdvancedVideoDataset(
    UCF101BaseVideoDataset, BaseAdvancedVideoDataset
):
    """
    Kinetics-600 advanced video dataset
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):
        if split == "validation":
            split = "test"
        
        self.augment_pipe = AugmentPipe(**cfg.augmentation)
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    def on_before_prepare_clips(self) -> None:
        self.setup()

    def setup(self) -> None:
        super().setup()

    def _augment(self, video: torch.Tensor) -> torch.Tensor:
        # augment video
        video, label = self.augment_pipe(video)
        return video

    def load_cond(
        self, video_idx: int
    ) -> torch.Tensor:
        return torch.tensor(
            self.metadata[video_idx]['labels'], dtype=torch.long
        ).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video_idx, clip_idx = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        start_frame, end_frame = clip_idx, min(clip_idx + self.n_frames, video_length)

        video, latent, cond = None, None, None
        if self.use_preprocessed_latents:
            latent = self.load_latent(video_metadata, start_frame, end_frame)

        # do not load video if we are training with latents
        if self.use_preprocessed_latents and self.split == "training":
            if self.external_cond_dim > 0:
                cond = self.load_cond(video_idx)
        else:
            if self.external_cond_dim > 0:
                # load video together with condition
                video = self.load_video(video_metadata, start_frame, end_frame)
                cond = self.load_cond(video_idx)
            else:
                # load video only
                video = self.load_video(video_metadata, start_frame, end_frame)

        lens = [len(x) for x in (video, latent) if x is not None]
        assert len(set(lens)) == 1, "video, latent must have the same length"
        pad_len = self.n_frames - lens[0]

        nonterminal = torch.ones(self.n_frames, dtype=torch.bool)
        if pad_len > 0:
            if video is not None:
                video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            if latent is not None:
                latent = F.pad(latent, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            if cond is not None:
                cond = F.pad(cond, (0, 0, 0, pad_len)).contiguous()
            nonterminal[-pad_len:] = 0

        if self.frame_skip > 1:
            if video is not None:
                video = video[:: self.frame_skip]
            if latent is not None:
                latent = latent[:: self.frame_skip]
            nonterminal = nonterminal[:: self.frame_skip]
        if cond is not None:
            cond = self._process_external_cond(cond)

        augmented_video = self._augment(video) if video is not None else None
        
        output = {
            "videos": self.transform(video) if video is not None else None,
            "augmented_videos": self.transform(augmented_video) if augmented_video is not None else None,
            "latents": latent,
            "conds": cond,
            "nonterminal": nonterminal,
        }
        return {key: value for key, value in output.items() if value is not None}