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
from algorithms import DFoTVideo, ReferenceDFoTVideo
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
    DMLabAdvancedVideoDataset
)
from datetime import datetime
import time
from torchsummary import summary

from utils.torch_utils import freeze_model
from utils.logging_utils import log_video
from accelerate import Accelerator
# from accelerate.utils.other import TORCH_SAFE_GLOBALS
# import omegaconf

# torch.serialization.add_safe_globals(TORCH_SAFE_GLOBALS)
# torch.serialization.add_safe_globals([omegaconf.ListConfig, omegaconf.base.ContainerMetadata, Any])
from lightning.pytorch.callbacks import ModelSummary

rank_zero_print = print

class SimpleVideoGenerationExperiment:
    compatible_algorithms = dict(
        dfot_video=DFoTVideo,
        reference_dfot_video=ReferenceDFoTVideo
    )
    compatible_datasets=dict(
        # video datasets
        minecraft=MinecraftAdvancedVideoDataset,
        realestate10k=RealEstate10KAdvancedVideoDataset,
        realestate10k_ood=RealEstate10KOODAdvancedVideoDataset,
        realestate10k_mini=RealEstate10KMiniAdvancedVideoDataset,
        kinetics_600=Kinetics600AdvancedVideoDataset,
        ucf_101=UCF101AdvancedVideoDataset,
        cond_ucf_101=UCF101AdvancedVideoDataset,
        cond_ucf_101_scaling=UCF101AdvancedVideoDataset,
        split_cond_ucf_101=SplitUCF101AdvancedVideoDataset,
        bair=BAIRAdvancedVideoDataset,
        dmlab=DMLabAdvancedVideoDataset
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
        self.tasks = [
            task for task in ["prediction", "interpolation"] \
            if self.cfg.algorithm.tasks[task].enabled 
        ]
        self.data_module: BaseDataModule = self._build_data_module()
        self.model: pl.LightningModule = self._build_algo()
        
        self.global_step = 0

        # temp
        self.barebones = False
        self.num_logged_videos = 0

        self.ema: EMAModel = None

    def _build_algo(self) -> pl:
        algo_name = self.cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(f"Algorithm {algo_name} is not found.")

        algo_class = self.compatible_algorithms[algo_name]
        model = algo_class(self.cfg.algorithm)
        model._trainer = self
        return model

    def _build_ema_model(self):
        if self.cfg.experiment.ema.enable:
            self.ema = EMAModel(self.model, decay=self.cfg.experiment.ema.decay)

    def _build_metrics(self, device):
        registry = SharedVideoMetricModelRegistry()
        metrics = {}
        metric_types = self.cfg.algorithm.logging.metrics
        for task in self.tasks:
            match task:
                case "prediction":
                    metrics_prediction = VideoMetric(
                        registry,
                        metric_types,
                        split_batch_size=self.cfg.algorithm.logging.metrics_batch_size,
                    )
                    metrics_prediction = metrics_prediction.to(device)
                    # freeze the metrics model
                    freeze_model(metrics_prediction)
                    metrics["prediction"] = metrics_prediction
                case "interpolation":
                    assert (
                        not self.model.use_causal_mask
                        and not self.model.is_full_sequence
                        and self.model.max_tokens > 2
                    ), "To execute interpolation, the model must be non-causal, not full sequence, and be able to process more than 2 tokens."
                    metrics_interpolation = VideoMetric(
                        registry,
                        metric_types,
                        split_batch_size=self.cfg.algorithm.logging.metrics_batch_size,
                    )
                    metrics_interpolation = metrics_interpolation.to(device)
                    freeze_model(metrics_interpolation)
                    metrics["interpolation"] = metrics_interpolation
                case _:
                    raise ValueError(f"Task {task} is not supported.")
        return metrics

    def _build_data_module(self) -> BaseDataModule:
        return BaseDataModule(self.cfg, self.compatible_datasets)

    def training(self) -> None:
        self.data_module.setup("fit")
        train_loader: DataLoader = self.data_module.train_dataloader()
        val_loader: DataLoader = self.data_module.val_dataloader()

        optimizer_cfg: Optimizer = self.model.configure_optimizers()
        lr_scheduler =  optimizer_cfg.pop("lr_scheduler", None)['scheduler']
        optimizer: Optimizer = optimizer_cfg.pop("optimizer", None)

        gradient_clip_val = self.cfg.experiment.training.optim.gradient_clip_val

        # accelerator
        accelerator = Accelerator(mixed_precision=self.cfg.experiment.training.precision)
        accelerator.register_for_checkpointing(lr_scheduler)
        device = accelerator.device

        self.metrics = self._build_metrics(device=device)
        self.model, optimizer, train_loader, val_loader = accelerator.prepare(
            self.model, optimizer, train_loader, val_loader
        )
        train_yielder = make_data_yielder(train_loader)

        # EMA after model and optimizer are prepared
        self._build_ema_model()
        if self.ema:
            rank_zero_print("Using EMA model with decay:", self.cfg.experiment.ema.decay)

        # frequency
        # logging
        log_grad_norm_freq = self.cfg.algorithm.logging.grad_norm_freq
        log_loss_freq = self.cfg.algorithm.logging.loss_freq
        # validation
        val_freq = self.cfg.experiment.validation.val_every_n_step
        # checkpointing
        checkpointing_freq = self.cfg.experiment.training.checkpointing.every_n_train_steps
        save_top_k = self.cfg.experiment.training.checkpointing.save_top_k
        if checkpointing_freq is None:
            warnings.warn(
                "Checkpointing frequency is not set. Checkpointing will not be performed during training.",
                UserWarning,
            )
        
        params = self.model.get_params()

        start_time = time.time()
        rank_zero_print("********** Starting training... **********")
        rank_zero_print("Configuration:", self.cfg)
        rank_zero_print("Model:", self.model)
        # Log parameters
        for p in params['trainable']:
            rank_zero_print(f"Trainable parameter: {p}")
        for p in params['frozen']:
            rank_zero_print(f"Frozen parameter: {p}")

        rank_zero_print("Total trainable parameters:", params['total_trainable']/ 1e6, "M")
        rank_zero_print("Total frozen parameters:", params['total_frozen'] / 1e6, "M")
        rank_zero_print("Total parameters:", params['total'] / 1e6, "M")

        rank_zero_print("Optimizer:", optimizer)
        rank_zero_print(f"Using device: {device}")
        if lr_scheduler:
            rank_zero_print("Learning Rate Scheduler:", lr_scheduler)
        rank_zero_print("Dataset:", self.data_module.root_cfg.dataset._name)
        rank_zero_print("Num training steps:", self.cfg.experiment.training.max_steps)
        rank_zero_print("Num examples:", len(train_loader.dataset))
        rank_zero_print("Num batches:", len(train_loader))

        self.global_step = 1
        if self.ckpt_path:
            rank_zero_print(f"Resuming from checkpoint: {self.ckpt_path}")
            self.global_step = self.resume_checkpoint(accelerator, self.ckpt_path)
            rank_zero_print(f"Resumed global step: {self.global_step}")

        while self.global_step < self.cfg.experiment.training.max_steps+1:
            self.model.train()
            batch = next(train_yielder)

            # Preprocess
            # TODO: scaling data takes a lot of time, consider moving it to the dataset
            batch = self.model.on_after_batch_transfer(batch, self.global_step)

            # Training step
            loss_dict = self.model.training_step(batch, self.global_step)
            loss = loss_dict["loss"]
            accelerator.backward(loss)

            # Log loss and lr
            if log_loss_freq and self.global_step % log_loss_freq == 0:
                 if self.logger:
                    self.logger.log_metrics({"training/loss": loss}, step=self.global_step)
                    self.logger.log_metrics({"training/lr": optimizer.param_groups[0]["lr"]}, step=self.global_step)
                 rank_zero_print(f"Step: {self.global_step}, eta: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}, "
                                 f"loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

            # Log max gradient norm before clipping
            if accelerator.is_main_process and log_grad_norm_freq and self.global_step % log_grad_norm_freq == 0:
                max_grad_norm = 0.0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        v_grad_norm = param.grad.data.norm(2).item()
                        max_grad_norm = max(max_grad_norm, v_grad_norm)
                if self.logger:
                    self.logger.log_metrics({"training/max_grad_norm": max_grad_norm}, step=self.global_step)
                rank_zero_print(f"Step: {self.global_step}, max_grad_norm: {max_grad_norm:.4f}")

            # Clip gradients if specified
            if gradient_clip_val is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.model.parameters(), gradient_clip_val)

            # Log gradidents
            if accelerator.is_main_process and self.logger and log_grad_norm_freq and self.global_step % log_grad_norm_freq == 0:
                norms = grad_norm(self.model.diffusion_model, norm_type=2)
                self.log_dict(norms, None)

            # Optimization step
            optimizer.step()
            lr_scheduler.step() if lr_scheduler else None
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                # EMA update
                if self.ema:
                    self.ema.step(accelerator.unwrap_model(self.model))

                # Validation step
                if val_freq and self.global_step % val_freq == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        self._validate(val_loader, accelerator)

                # Save checkpoints
                if checkpointing_freq and self.global_step % checkpointing_freq == 0:
                    self.save_checkpoint(self.global_step, accelerator, save_top_k)

                self.global_step += 1

    def validation(self):
        assert self.ckpt_path, "Checkpoint path must be provided for validation."

        self.data_module.setup("validate")
        val_loader: DataLoader = self.data_module.val_dataloader()

        # accelerator
        accelerator = Accelerator(mixed_precision=self.cfg.experiment.training.precision)
        device = accelerator.device

        self.metrics = self._build_metrics(device=device)
        self.model, val_loader = accelerator.prepare(self.model, val_loader)

        # EMA after model and optimizer are prepared
        # check if EMA checkpoint exists
        if Path(os.path.join(self.ckpt_path, "ema.safetensors")).exists():
            self._build_ema_model()
        self.resume_checkpoint(accelerator, self.ckpt_path)
        self._validate(val_loader, accelerator)
        

    def _validate(self, val_loader, accelerator, namespace='validation') -> None:
        # Load EMA if enabled
        if self.ema:
            self.model = accelerator.unwrap_model(self.model)
            self.ema.store(self.model)
            self.ema.copy_to(self.model)

        val_yielder = make_data_yielder(val_loader)

        # Load required components for validation
        self.model.new_on_validation_epoch_start()
        num_validate_batches = min(len(val_loader), self.cfg.experiment.validation.limit_batch)

        rank_zero_print("\n********** Starting validation... **********")
        rank_zero_print("Num batches:", num_validate_batches)

        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for i, batch in enumerate(val_yielder):
                if i >= num_validate_batches:
                    break

                # Preprocess
                batch = self.model.on_after_batch_transfer(batch, i)
                denoising_output, all_videos = self.model.new_validation_step(batch, i, namespace="validation")

                # Process denoising output
                total_loss += denoising_output["loss"].item()
                gt_videos = denoising_output["gts"]
                recons = denoising_output["recons"]
                num_videos_to_log = min(self.cfg.algorithm.logging.max_num_videos - self.num_logged_videos, gt_videos.shape[0])
                if self.logger:
                    log_video(
                        recons[:num_videos_to_log],
                        gt_videos[:num_videos_to_log],
                        step=self.global_step,
                        namespace=f"{namespace}_denoising_vis",
                        logger=self.logger.experiment,
                        indent=self.num_logged_videos,
                        captions="denoised | gt",
                    )
    
                # Log samples
                if self.logger:
                    self.log_videos(all_videos, namespace=namespace, context_frames=self.model.n_context_frames, global_step=i)
                
                accelerator.wait_for_everyone()
                #TODO: gather distributed data if needed
                # Update metrics
                self.update_metrics(all_videos)
        
        if accelerator.is_main_process:
            # Log average loss
            avg_loss = total_loss / num_validate_batches
            rank_zero_print(f"Validation average loss: {avg_loss:.4f}")
            if self.logger:
                self.logger.log_metrics({f"{namespace}/avg_loss": avg_loss}, step=self.global_step)
            
            # compute metrics
            for task in self.tasks:
                result = self.metrics[task].log(task)
                rank_zero_print(f"Validation {task} metrics:")
                # Log metrics
                self.log_dict(result, namespace, is_print=True)

        # Remove unnecessary components
        self.model.generator = None
        self.model.vae = None
        self.model.num_logged_videos = 0
        self.num_logged_videos = 0
        self.model.train()

        # Restore EMA if it was used
        if self.ema:
            # Restore original model parameters
            self.ema.restore(self.model)
            self.model = accelerator.prepare(self.model)

        rank_zero_print("********** Validation completed **********\n")

    def update_metrics(self, all_videos) -> None:
        gt_videos = all_videos["gt"]
        for task in self.tasks:
            metric = self.metrics[task]
            videos = all_videos[task]
            context_mask = torch.zeros(videos.shape[1]).bool().to(videos.device)
            match task:
                case "prediction":
                    context_mask[: self.model.n_context_frames] = True
                case "interpolation":
                    context_mask[[0, -1]] = True
            if self.model.logging.n_metrics_frames is not None:
                context_mask = context_mask[: self.model.logging.n_metrics_frames]
            metric(videos, gt_videos, context_mask=context_mask)

    def log_dict(self, dictionary: Dict[str, Union[float, torch.Tensor]], namespace: str, is_print=False) -> None:
        """Log a dictionary of values to the logger."""
        if not is_rank_zero:
            return
        for key, value in dictionary.items():
            if is_print:
                key = f"{namespace}/{key}" if namespace else key
                rank_zero_print(f"{key}: {value:.4f}")

            if self.logger:
                self.logger.log_metrics(
                    {key: value},
                    step=self.global_step,
                )

    def log_videos(self, all_videos: Dict[str, torch.Tensor], namespace: str, context_frames=None, global_step=None) -> None:
        """Log videos during validation/test step."""
        # TODO: gather distributed data if needed
        batch_size, n_frames = all_videos["gt"].shape[:2]
        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.cfg.algorithm.logging.max_num_videos
        ):
            return

        num_videos_to_log = min(
            self.cfg.algorithm.logging.max_num_videos - self.num_logged_videos,
            batch_size,
        )
        cut_videos = lambda x: x[:num_videos_to_log]

        if context_frames is None:
            context_frames=self.model.n_context_frames if task == "prediction" else torch.tensor([0, n_frames - 1], device=all_videos["gt"].device, dtype=torch.long)


        for task in self.tasks:
            # (f"{namespace}_{task}_vis")
            log_video(
                cut_videos(all_videos[task]),
                cut_videos(all_videos["gt"]),
                step=None if namespace == "test" else global_step,
                namespace=f"{namespace}_{task}_vis",
                logger=self.logger.experiment,
                indent=self.num_logged_videos,
                raw_dir=self.cfg.algorithm.logging.raw_dir,
                context_frames=context_frames,
                captions=f"{task} | gt",
            )

        self.num_logged_videos += batch_size

    def resume_checkpoint(self, accelerator: Accelerator, ckpt_path: str) -> None:
        if not ckpt_path or not Path(ckpt_path).exists():
            rank_zero_print("No checkpoint found. Starting from scratch.")
            return

        rank_zero_print(f"Resuming from checkpoint: {ckpt_path}")
        # Load pytorch lightning state
        if ckpt_path.endswith(".ckpt"):
            state_dict = torch.load(ckpt_path, map_location=accelerator.device, weights_only=False)
            # EMA
            self.model.on_load_checkpoint(state_dict)
            self.model = accelerator.unwrap_model(self.model)
            self.model.load_state_dict(state_dict['state_dict'], strict=False)
            self.model = accelerator.prepare(self.model)
            return state_dict['global_step']
        
        # TODO: fix weights_only for safer
        accelerator.load_state(ckpt_path, load_kwargs={'weights_only': False})
        global_step = int(os.path.basename(ckpt_path).split('_')[1].split('.')[0])
        
        if self.ema:
            ema_save_path = os.path.join(ckpt_path, "ema.safetensors")
            if Path(ema_save_path).exists():
                rank_zero_print(f"Loading EMA state from {ema_save_path}")
                self.ema.load_state_dict(load_file(ema_save_path))
            else:
                rank_zero_print(f"No EMA state found at {ema_save_path}. Skipping EMA loading.")
        return global_step

    def save_checkpoint(self, global_step: int, accelerator: Accelerator, save_top_k=None) -> None:
        save_dir = os.path.join(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"], "checkpoints")
        save_path = os.path.join(save_dir, f"checkpoint_{global_step}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if save_top_k:
            checkpoints = os.listdir(save_dir)
            checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[1]))
            if len(checkpoints) >= save_top_k:
                num_to_remove = len(checkpoints) - save_top_k + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                rank_zero_print(f'{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints: {removing_checkpoints}')

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        rank_zero_print(f"Saving checkpoint to {save_path}")

        # Save the model state
        accelerator.save_state(save_path)

        # Save the EMA state if applicable
        if self.ema:
            ema_save_path = os.path.join(save_path, "ema.safetensors")
            save_file(self.ema.state_dict(), ema_save_path)
            rank_zero_print(f"EMA state saved to {ema_save_path}")

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


def make_data_yielder(dataloader):
    while True:
        for batch in dataloader:
            yield batch