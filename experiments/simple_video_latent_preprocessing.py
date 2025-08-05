import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from typing import Optional, Union, Dict, Any
from utils.distributed_utils import rank_zero_print, is_rank_zero
from pathlib import Path
import shutil
import os
import hydra
from safetensors.torch import save_file, load_file
from utils.lightning_utils import EMA
from lightning.pytorch.loggers.wandb import WandbLogger
import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm
from .data_modules import BaseDataModule
import warnings
from algorithms import DFoTVideo, DifferenceDFoTVideo
from algorithms.common.metrics.video import VideoMetric, SharedVideoMetricModelRegistry
from algorithms.common.ema import EMAModel
from datasets.video import (
    MinecraftAdvancedVideoDataset,
    Kinetics600AdvancedVideoDataset,
    RealEstate10KAdvancedVideoDataset,
    RealEstate10KMiniAdvancedVideoDataset,
    RealEstate10KOODAdvancedVideoDataset,
    UCF101AdvancedVideoDataset,
    SplitUCF101AdvancedVideoDataset,
    BAIRAdvancedVideoDataset,
    DMLabAdvancedVideoDataset,
    TaichiAdvancedVideoDataset
)
import gc
import datetime
import time
from torchsummary import summary

from utils.torch_utils import freeze_model
from utils.logging_utils import log_video
from accelerate import Accelerator, PartialState, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from lightning.pytorch.callbacks import ModelSummary
from datasets.video import (
    MinecraftSimpleVideoDataset,
    UCF101SimpleVideoDataset,
    DMLabSimpleVideoDataset,
    BAIRSimpleVideoDataset,
    TaichiSimpleVideoDataset
)
from algorithms.vae import ImageVAEPreprocessor, DCAEPreprocessor, AutoencoderKL, AutoencoderKLPreprocessor, Titok_KLPreprocessor, NewTitok_KLPreprocessor
from .base_exp import BaseLightningExperiment
from .data_modules import ValDataModule
#os.path.append('..')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True
# torch.use_deterministic_algorithms(True)
# from test.taichi import test_func

rank_zero_print = print
logger = get_logger(__name__)

class SimpleVideoLatentPreprocessingExperiment:
    compatible_algorithms = dict(
        image_vae_preprocessor=ImageVAEPreprocessor,
        dc_ae_preprocessor=DCAEPreprocessor,
        dc_ae_16x_preprocessor=DCAEPreprocessor,
        kl_autoencoder_preprocessor=AutoencoderKLPreprocessor,
        titok_kl_preprocessor=Titok_KLPreprocessor,
        new_titok_kl_preprocessor=NewTitok_KLPreprocessor,
    )
    compatible_datasets=dict(
        # video datasets
        minecraft=MinecraftSimpleVideoDataset,
        ucf_101=UCF101SimpleVideoDataset,
        dmlab=DMLabSimpleVideoDataset,
        bair=BAIRSimpleVideoDataset,
        taichi=TaichiSimpleVideoDataset,
        test_taichi=TaichiSimpleVideoDataset,
    )
    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, Path]] = None,
    ):
        self.cfg = root_cfg
        self.logger = logger

        self.ckpt_path = ckpt_path
        self.num_logged_videos = 0

    def _build_algo(self) -> pl:
        algo_name = self.cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(f"Algorithm {algo_name} is not found.")

        algo_class = self.compatible_algorithms[algo_name]
        model = algo_class(self.cfg.algorithm)
        model._trainer = self
        return model

    def _build_data_module(self) -> ValDataModule:
        return ValDataModule(self.cfg, self.compatible_datasets)

    def validation(self):
        """Run validation step for the experiment.
           Pass ema checkpoint if you want to use EMA model for validation.
           If you want to use the model from the checkpoint, pass the path to the checkpoint.
        """
        self.model: pl.LightningModule = self._build_algo()
        self.data_module: BaseDataModule = self._build_data_module()
        self.data_module.setup("validate")
        val_loaders: DataLoader = self.data_module.val_dataloader()

        # accelerator
        accelerator = Accelerator(
            mixed_precision='no'
        )
        # test_func()

        # Seed
        if self.cfg.experiment.validation.manual_seed is not None:
            set_seed(self.cfg.experiment.validation.manual_seed, device_specific=True)

        self.model = accelerator.prepare(self.model)
        # logger.info(f"  Model: {self.model}")
        if not isinstance(val_loaders, list):
            val_loaders = [val_loaders]
        
        for i, val_loader in enumerate(val_loaders):
            # prepare the dataloader
            logger.info(f"Preparing dataloader {self.cfg.experiment.validation.dataset_splits[i]}...")
            val_loader = accelerator.prepare(val_loader)
            self.run_validation(val_loader)

        

    def run_validation(self, val_loader) -> None:
        # Load required components for validation
        logger.info(f'len(val_loader) {len(val_loader)}', main_process_only=False)
        num_validate_batches = len(val_loader)
        if num_validate_batches == 0:
            logger.warning("Validation dataloader is empty. Skipping validation.", main_process_only=False)
            return

        logger.info("********** Starting validation... **********", main_process_only=False)
        logger.info(f"Num batches: {num_validate_batches}", main_process_only=False)

        # total_batch = 0
        self.model = self.model.to('cuda')
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):                
                if (i + 1) % 10 == 0:
                    logger.info(f"Validation batch {i + 1}/{num_validate_batches}...", main_process_only=False)
                self.model.validation_step(batch, i)

        self.num_logged_videos = 0
        # self.model.eva()

        logger.info("********** Validation completed **********\n")

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            # rank_zero_print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )
