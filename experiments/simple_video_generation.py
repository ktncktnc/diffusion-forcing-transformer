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
import omegaconf

from utils.torch_utils import freeze_model
from utils.logging_utils import log_video
from accelerate import Accelerator, PartialState, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from lightning.pytorch.callbacks import ModelSummary

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


rank_zero_print = print
logger = get_logger(__name__)

class SimpleVideoGenerationExperiment:
    compatible_algorithms = dict(
        dfot_video=DFoTVideo,
        difference_dfot_video=DifferenceDFoTVideo,
        # reference_dfot_video=ReferenceDFoTVideo
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
        dmlab=DMLabAdvancedVideoDataset,
        taichi=TaichiAdvancedVideoDataset,
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

    def _build_ema_model(self, model: Optional[torch.nn.Module] = None) -> None:
        model = model or self.model
        if self.cfg.experiment.ema.enable:
            self.ema = EMAModel(model, decay=self.cfg.experiment.ema.decay)

    def _build_metrics(self, device):
        registry = SharedVideoMetricModelRegistry()
        metrics = {}
        metric_types = self.cfg.algorithm.logging.metrics
        for task in self.tasks:
            match task:
                case "prediction":
                    # NOTE: disable sync_on_compute because it will cause OOM error when using torchmetrics with accelerator
                    metrics_prediction = VideoMetric(
                        registry,
                        metric_types,
                        split_batch_size=self.cfg.algorithm.logging.metrics_batch_size,
                        torchmetrics_kwargs={'sync_on_compute': False}
                    )
                    metrics_prediction = metrics_prediction.to(device)
                    # freeze the metrics model
                    freeze_model(metrics_prediction)
                    metrics["prediction"] = metrics_prediction
                case "interpolation":
                    assert (
                        not get_org_model(self.model).use_causal_mask
                        and not get_org_model(self.model).is_full_sequence
                        and get_org_model(self.model).max_tokens > 2
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
        from accelerate.utils import InitProcessGroupKwargs
        timeout = InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=15))
        accelerator = Accelerator(
            mixed_precision=self.cfg.experiment.training.precision,
            gradient_accumulation_steps=self.cfg.experiment.training.optim.accumulate_grad_batches,
            kwargs_handlers=[timeout],
        )
        logger.info(f"  Configuration: {omegaconf.OmegaConf.to_yaml(self.cfg)}")

        self.model: pl.LightningModule = self._build_algo()

        self.data_module: BaseDataModule = self._build_data_module()
        self.data_module.setup("fit")
        train_loader: DataLoader = self.data_module.train_dataloader()
        val_loader: DataLoader = self.data_module.val_dataloader()

        optimizer_cfg: Optimizer = self.model.configure_optimizers()
        lr_scheduler =  optimizer_cfg.pop("lr_scheduler", None)['scheduler']
        optimizer: Optimizer = optimizer_cfg.pop("optimizer", None)
        gradient_clip_val = self.cfg.experiment.training.optim.gradient_clip_val

        # accelerator
        device = accelerator.device

        # Seed
        if self.cfg.experiment.training.manual_seed is not None:
            set_seed(self.cfg.experiment.training.manual_seed, device_specific=True)

        self.metrics = self._build_metrics(device=device)
        self.model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
            self.model, optimizer, train_loader, val_loader, lr_scheduler
        )
        train_yielder = make_infinite_loader(train_loader)

        # EMA after model and optimizer are prepared
        self._build_ema_model(accelerator.unwrap_model(self.model))
        if self.ema:
            logger.info(f"Using EMA model with decay: {self.cfg.experiment.ema.decay}")

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
        
        params = get_org_model(self.model).get_params()

        start_time = time.time()
        logger.info("********** Starting training... **********")
        logger.info(f"  Model: {self.model}")
        # Log parameters
        logger.info(f"  Total trainable parameters: {params['total_trainable']:,d}")
        logger.info(f"  Total frozen parameters: {params['total_frozen']:,d}")
        logger.info(f"  Total parameters: {params['total']:,d}")
        logger.info(f'  Model x_shape: {get_org_model(self.model).x_shape}')
        logger.info(f"  Model max_tokens: {get_org_model(self.model).max_tokens}")
        logger.info(f"  Model external_cond_type: {get_org_model(self.model).external_cond_type}")
        logger.info(f"  Model external_cond_num_classes: {get_org_model(self.model).external_cond_num_classes}")
        logger.info(f"  Model external_cond_dim: {get_org_model(self.model).external_cond_dim}")

        state = PartialState()
        logger.info(f"  Distributed type: {state.distributed_type}")
        logger.info(f"  Number of processes: {state.num_processes}")
        logger.info(f"  Local process index: {state.local_process_index}")

        logger.info(f"  Optimizer: {optimizer}")
        logger.info(f"  Using {accelerator.num_processes} GPUs")
        logger.info(f"  Using device: {device}")

        if lr_scheduler:
            logger.info(f"  Learning Rate Scheduler {lr_scheduler}")
        logger.info(f"  Dataset: {self.data_module.root_cfg.dataset._name}")
        logger.info(f"  Num training steps {self.cfg.experiment.training.max_steps}")
        logger.info(f"  Batch size: {self.cfg.experiment.training.batch_size}")
        logger.info(f"  Num examples: {len(train_loader.dataset)}")
        logger.info(f"  Num batches: {len(train_loader)}")
        logger.info(f"  Num workers: {self.data_module._get_num_workers(self.cfg.experiment.training.data.num_workers)}")
        logger.info(f"  Gradient clipping value: {gradient_clip_val}")

        self.global_step = 1
        if self.ckpt_path:
            logger.info(f"Resuming from checkpoint: {self.ckpt_path}")
            self.global_step = self.resume_checkpoint(accelerator, self.ckpt_path) + 1
            logger.info(f"Resumed global step: {self.global_step}")

        logger.info("********** Starting training... **********")
        while self.global_step < self.cfg.experiment.training.max_steps+1:
            self.model.train()
            batch = next(train_yielder)
            # Preprocess
            # TODO: scaling data takes a lot of time, consider moving it to the dataset
            batch = get_org_model(self.model).on_after_batch_transfer(batch, self.global_step)

            # Training step
            with accelerator.accumulate(self.model):
                loss_dict = get_org_model(self.model).training_step(batch, self.global_step)
                loss = loss_dict["loss"]
                accelerator.backward(loss)

                # Clip gradients if specified
                if gradient_clip_val is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.model.parameters(), gradient_clip_val)

                # Optimization step
                optimizer.step()
                lr_scheduler.step() if lr_scheduler else None
                optimizer.zero_grad()

            # Log loss and lr
            if accelerator.is_main_process and log_loss_freq and self.global_step % log_loss_freq == 0:
                if self.logger:
                    log_loss_dict = {'training/loss': loss.item()}
                    if 'diff_loss' in loss_dict:
                        log_loss_dict['training/diff_loss'] = loss_dict.pop('diff_loss')
                    if 'xs_loss' in loss_dict:
                        log_loss_dict['training/xs_loss'] = loss_dict.pop('xs_loss')
                    self.logger.log_metrics(log_loss_dict, step=self.global_step)
                    self.logger.log_metrics({"training/lr": optimizer.param_groups[0]["lr"]}, step=self.global_step)
                logger.info(f"Step: {self.global_step}, eta: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}, "f"loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

            # Log max gradient norm before clipping
            if accelerator.is_main_process and log_grad_norm_freq and self.global_step % log_grad_norm_freq == 0:
                max_grad_norm = 0.0
                for name, param in get_org_model(self.model).named_parameters():
                    if param.grad is not None:
                        v_grad_norm = param.grad.data.norm(2).item()
                        max_grad_norm = max(max_grad_norm, v_grad_norm)
                if self.logger:
                    self.logger.log_metrics({"training/max_grad_norm": max_grad_norm}, step=self.global_step)
                logger.info(f"Step: {self.global_step}, max_grad_norm: {max_grad_norm:.4f}")

            # Log gradidents
            if accelerator.is_main_process and self.logger and log_grad_norm_freq and self.global_step % log_grad_norm_freq == 0:
                norms = grad_norm(get_org_model(self.model).diffusion_model, norm_type=2)
                self.log_dict(norms, None)

            if accelerator.sync_gradients:
                # EMA update
                if self.ema:
                    self.ema.step(accelerator.unwrap_model(self.model))

                # Save checkpoints 
                if checkpointing_freq and self.global_step % checkpointing_freq == 0:
                    self.save_checkpoint(self.global_step, accelerator, save_top_k)

                # Validation step
                if val_freq and self.global_step % val_freq == 0:
                    accelerator.wait_for_everyone()
                    # if accelerator.is_main_process:
                    self.run_validation(val_loader, accelerator)

            self.global_step += 1

        # save final checkpoint
        self.save_checkpoint(self.global_step, accelerator, save_top_k)
        logger.info("********** Training completed **********\n")

    def validation(self):
        """Run validation step for the experiment.
            Pass ema checkpoint if you want to use EMA model for validation.
            If you want to use the model from the checkpoint, pass the path to the checkpoint.
        """
        assert self.ckpt_path, "Checkpoint path must be provided for validation."
        self.model: pl.LightningModule = self._build_algo()
        self.data_module: BaseDataModule = self._build_data_module()
        self.data_module.setup("validate")
        val_loader: DataLoader = self.data_module.val_dataloader()

        # accelerator
        accelerator = Accelerator(mixed_precision=self.cfg.experiment.training.precision)
        device = accelerator.device

        # Seed
        if self.cfg.experiment.validation.manual_seed is not None:
            set_seed(self.cfg.experiment.validation.manual_seed, device_specific=True)

        self.metrics = self._build_metrics(device=device)
        self.model, val_loader = accelerator.prepare(self.model, val_loader)
        self.resume_checkpoint(accelerator, self.ckpt_path)
        params = get_org_model(self.model).get_params()

        logger.info(f"  Model: {self.model}")
        logger.info(f"  Total trainable parameters: {params['total_trainable']:,d}")
        logger.info(f"  Total frozen parameters: {params['total_frozen']:,d}")
        logger.info(f"  Total parameters: {params['total']:,d}")
        logger.info(f"Sampling step: {self.cfg.algorithm.diffusion.sampling_timesteps}")
        logger.info(f"Scheduling matrix: {self.cfg.algorithm.scheduling_matrix}")
        self.run_validation(val_loader, accelerator)
        

    def run_validation(self, val_loader, accelerator, validate_ema=True, namespace='validation') -> None:
        # Load EMA if enabled
        val_yielder = make_infinite_loader(val_loader)
        model = accelerator.unwrap_model(self.model)
        if validate_ema and self.ema:
            self.ema.store(model)
            self.ema.copy_to(model)

        max_num_videos = self.cfg.algorithm.logging.max_num_videos // accelerator.num_processes
        # Load required components for validation
        model.new_on_validation_epoch_start()
        logger.info(f'len(val_loader) {len(val_loader)}', main_process_only=True)
        num_validate_batches = min(len(val_loader), self.cfg.experiment.validation.limit_batch)

        logger.info("********** Starting validation... **********", main_process_only=True)
        logger.info(f"  Batch size: {self.cfg.experiment.validation.batch_size}", main_process_only=True)
        logger.info(f"  Num batches: {num_validate_batches}", main_process_only=True)

        # total_batch = 0
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_diff_loss = 0.0
            total_xs_loss = 0.0
            for i, batch in enumerate(val_yielder):
                if i >= num_validate_batches:
                    break
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Validation batch {i + 1}/{num_validate_batches}...", main_process_only=False)
                # Preprocess
                batch = model.on_after_batch_transfer(batch, i)
                denoising_output, all_videos = model.new_validation_step(batch, i, accelerator, namespace="validation")

                loss_scalar = denoising_output["loss"].mean()
                
                # Gather only the scalar losses (very small memory footprint)
                gathered_loss = accelerator.gather_for_metrics(loss_scalar)
                gathered_diff_loss = accelerator.gather_for_metrics(denoising_output.get("diff_loss", torch.tensor(0.0)).mean())  if "diff_loss" in denoising_output else torch.tensor(0.0)
                gathered_xs_loss = accelerator.gather_for_metrics(denoising_output.get("xs_loss", torch.tensor(0.0)).mean()) if "xs_loss" in denoising_output else torch.tensor(0.0)
                
                # Accumulate losses on ALL processes
                total_loss += gathered_loss.mean().item()
                if gathered_diff_loss.sum().item() > 0:
                    total_diff_loss += gathered_diff_loss.mean().item()
                if gathered_xs_loss.sum().item() > 0:
                    total_xs_loss += gathered_xs_loss.mean().item()
                
                gathered_videos = accelerator.gather_for_metrics(all_videos)

                if accelerator.is_main_process:
                    gt_videos = denoising_output["gts"]
                    recons = denoising_output["recons"]
                    num_videos_to_log = min(max_num_videos - self.num_logged_videos, gt_videos.shape[0])

                    if self.logger and num_videos_to_log > 0:
                        log_video(
                            recons[:num_videos_to_log],
                            gt_videos[:num_videos_to_log],
                            step=self.global_step,
                            namespace=f"{namespace}_denoising_vis",
                            logger=self.logger.experiment,
                            indent=self.num_logged_videos,
                            captions="denoised | gt",
                            resize_to=(64, 64),
                        )

                    # Log samples
                    if self.logger and num_videos_to_log > 0:
                        self.log_videos(all_videos, namespace=namespace, context_frames=model.n_context_frames, global_step=i)
                
                    # Update metrics
                    self.update_metrics(gathered_videos)

                del gathered_videos, all_videos, denoising_output
                gc.collect()
                torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            # Log average loss
            avg_loss = total_loss / num_validate_batches
            logger.info(f"Validation average loss: {avg_loss:.4f}")
            log_loss = {f"{namespace}/avg_loss": avg_loss}

            if total_diff_loss > 0:
                avg_diff_loss = total_diff_loss / num_validate_batches
                logger.info(f"Validation average diff loss: {avg_diff_loss:.4f}")
                log_loss[f"{namespace}/avg_diff_loss"] = avg_diff_loss

            if total_xs_loss > 0:
                avg_xs_loss = total_xs_loss / num_validate_batches
                logger.info(f"Validation average xs loss: {avg_xs_loss:.4f}")
                log_loss[f"{namespace}/avg_xs_loss"] = avg_xs_loss

            if self.logger:
                self.logger.log_metrics(log_loss, step=self.global_step)

            # compute metrics
            for task in self.tasks:
                logger.info(f"Validation {task} metrics:")
                result = self.metrics[task].log(task)
                # Log metrics
                self.log_dict(result, namespace, is_print=True)

        # Remove unnecessary components
        model.generator = None
        model.num_logged_videos = 0
        self.num_logged_videos = 0
        model.new_on_validation_epoch_end()
        model.train()

        # Restore EMA if it was used
        if validate_ema and self.ema:
            # Restore original model parameters
            self.ema.restore(model)

        logger.info("********** Validation completed **********\n")

    def update_metrics(self, all_videos) -> None:
        # only consider the first n_metrics_frames for evaluation
        if get_org_model(self.model).logging.n_metrics_frames is not None:
            all_videos = {
                k: v[:, : get_org_model(self.model).logging.n_metrics_frames] for k, v in all_videos.items()
            }

        gt_videos = all_videos["gt"]
        for task in self.tasks:
            metric = self.metrics[task]
            videos = all_videos[task]
            context_mask = torch.zeros(videos.shape[1]).bool().to(videos.device)
            match task:
                case "prediction":
                    context_mask[: get_org_model(self.model).n_context_frames] = True
                case "interpolation":
                    context_mask[[0, -1]] = True
            if get_org_model(self.model).logging.n_metrics_frames is not None:
                context_mask = context_mask[: get_org_model(self.model).logging.n_metrics_frames]
            metric(videos, gt_videos, context_mask=context_mask)

    def log_dict(self, dictionary: Dict[str, Union[float, torch.Tensor]], namespace: str, is_print=False) -> None:
        """Log a dictionary of values to the logger."""
        if not is_rank_zero:
            return
        for key, value in dictionary.items():
            if is_print:
                key = f"{namespace}/{key}" if namespace else key
                logger.info(f"{key}: {value:.4f}")

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
            context_frames=get_org_model(self.model).n_context_frames if task == "prediction" else torch.tensor([0, n_frames - 1], device=all_videos["gt"].device, dtype=torch.long)


        for task in self.tasks:
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
                resize_to=(64, 64)
            )
            if "gt_diff" in all_videos:
                log_video(
                    cut_videos(all_videos[task + "_diff"].clone()),
                    cut_videos(all_videos["gt_diff"].clone()),
                    step=None if namespace == "test" else global_step,
                    namespace=f"{namespace}_{task}_diff_vis",
                    logger=self.logger.experiment,
                    indent=self.num_logged_videos,
                    raw_dir=self.cfg.algorithm.logging.raw_dir,
                    context_frames=context_frames,
                    captions=f"{task}_diff | gt_diff",
                    resize_to=(64, 64)
                )

        self.num_logged_videos += batch_size

    def resume_checkpoint(self, accelerator: Accelerator, ckpt_path: str) -> None:
        if not ckpt_path or not Path(ckpt_path).exists():
            logger.info("No checkpoint found. Starting from scratch.")
            raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist.")

        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        # Load pytorch lightning state
        if os.path.isdir(ckpt_path):
            accelerator.load_state(ckpt_path, load_kwargs={'weights_only': False})
            global_step = int(os.path.basename(ckpt_path).split('_')[1].split('.')[0])
            
            if self.ema:
                ema_save_path = os.path.join(ckpt_path, "ema.safetensors")
                if Path(ema_save_path).exists():
                    logger.info(f"Loading EMA state from {ema_save_path}")
                    self.ema.load_state_dict(load_file(ema_save_path))
                else:
                    logger.info(f"No EMA state found at {ema_save_path}. Skipping EMA loading.")
            return global_step
        # Only load model state if ckpt_path is a file
        if ckpt_path.endswith(".ckpt"):
            state_dict = torch.load(ckpt_path, map_location=accelerator.device, weights_only=False)['state_dict']
        elif ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path)
        self.model = accelerator.unwrap_model(self.model)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = accelerator.prepare(self.model)
        return -1

    def save_checkpoint(self, global_step: int, accelerator: Accelerator, save_top_k=None) -> None:
        save_dir = os.path.join(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"], "checkpoints")
        save_path = os.path.join(save_dir, f"checkpoint_{global_step}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if save_top_k:
            checkpoints = os.listdir(save_dir)
            checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[1]))
            if len(checkpoints) > save_top_k:
                num_to_remove = len(checkpoints) - save_top_k + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                logger.info(f'{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints: {removing_checkpoints}')

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                    shutil.rmtree(removing_checkpoint)

        logger.info(f"Saving checkpoint to {save_path}")
        accelerator.save_state(save_path)

        # Save the EMA state if applicable
        if self.ema:
            ema_save_path = os.path.join(save_path, "ema.safetensors")
            save_file(self.ema.state_dict(), ema_save_path)
            logger.info(f"EMA state saved to {ema_save_path}")

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


def make_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def get_org_model(model: pl.LightningModule) -> pl.LightningModule:
    """
    Get the original model from the wrapped model.
    This is useful when you want to access the original model's methods or attributes.
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model