import warnings
from typing import Any, Dict
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.apply_func import apply_to_collection
from omegaconf import DictConfig
from accelerate import Accelerator
import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from functools import partial
from utils.torch_utils import bernoulli_tensor
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities import grad_norm
from utils.torch_utils import freeze_model
from typing import Optional, Any, Dict, Literal, Callable, Tuple
from algorithms.common.metrics.video import VideoMetric, SharedVideoMetricModelRegistry
from einops import rearrange, repeat, reduce
from utils.distributed_utils import rank_zero_print, is_rank_zero
from utils.logging_utils import log_video
from transformers import get_scheduler
from algorithms.common.attn_hook import register_hooks, clear_hooks, save_attention_maps, attn_maps

from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print
from .base_pytorch_algo import BasePytorchAlgo
from algorithms.vae import ImageVAE, VideoVAE, MyAutoencoderDC, AutoencoderKL, TiTok_KL


class BaseVideoAlgo(BasePytorchAlgo):
    """
    A base class for Pytorch algorithms using Pytorch Lightning.
    See https://lightning.ai/docs/pytorch/stable/starter/introduction.html for more details.
    """
    def __init__(self, cfg: DictConfig):
        # 1. Shape
        self.x_shape = list(cfg.x_shape)
        self.frame_skip = cfg.frame_skip
        self.chunk_size = cfg.chunk_size
        self.external_cond_type = cfg.external_cond_type
        self.external_cond_num_classes = cfg.external_cond_num_classes
        self.external_cond_dim = cfg.external_cond_dim * (
            cfg.frame_skip if cfg.external_cond_stack else 1
        )

        # 2. Latent
        self.is_latent_diffusion = cfg.latent.enabled
        self.is_latent_online = cfg.latent.type == "online"
        self.temporal_downsampling_factor = cfg.latent.downsampling_factor[0]
        self.is_latent_video_vae = self.temporal_downsampling_factor > 1
        if self.is_latent_diffusion:
            if cfg.latent.shape is not None:
                self.x_shape = cfg.latent.shape
            else:
                self.x_shape = [cfg.latent.num_channels] + [
                    d // cfg.latent.downsampling_factor[1] for d in self.x_shape[1:]
                ]
            if self.is_latent_video_vae:
                self.check_video_vae_compatibility(cfg)

        # 3. Diffusion
        self.use_causal_mask = cfg.diffusion.use_causal_mask
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise
        if "cum_snr_decay" in cfg.diffusion.loss_weighting:
            cfg.diffusion.loss_weighting.cum_snr_decay = (
                cfg.diffusion.loss_weighting.cum_snr_decay**cfg.frame_skip
            )
        self.is_full_sequence = (
            cfg.noise_level == "random_uniform"
            and not cfg.fixed_context.enabled
            and not cfg.variable_context.enabled
        )

        # 4. Logging
        self.logging = cfg.logging
        self.tasks = [
            task
            for task in ["prediction", "interpolation"]
            if getattr(cfg.tasks, task).enabled
        ]
        self.num_logged_videos = 0
        self.generator = None

        super().__init__(cfg)

    # ---------------------------------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------------------------------
    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        """
        Preprocess the batch before training/validation.
        Args:
            batch (Dict): The batch of data. Contains "videos" or "latents", (optional) "conditions", and "masks".
            dataloader_idx (int): The index of the dataloader.
        Returns:
            xs (Tensor, "B n_tokens *x_shape"): Tokens to be processed by the model.
            conditions (Optional[Tensor], "B n_tokens d"): External conditions for the tokens.
            masks (Tensor, "B n_tokens"): Masks for the tokens.
            gt_videos (Optional[Tensor], "B n_frames *x_shape"): Optional ground truth videos, used for validation in latent diffusion.
        """
        # 1. Tokenize the videos and optionally prepare the ground truth videos
        gt_videos = None
        if self.is_latent_diffusion:
            if self.is_latent_online:
                xs = self._encode(batch["videos"])
            else:
                xs = batch['latents']

            if "videos" in batch:
                gt_videos = batch["videos"]
        else:
            xs = batch["videos"]

        xs = self._normalize_x(xs)

        # 2. Prepare external conditions
        conditions = batch.get("conds", None)

        # 3. Prepare the masks
        if "masks" in batch:
            assert (
                not self.is_latent_video_vae
            ), "Masks should not be provided from the dataset when using VideoVAE."
        else:
            masks = torch.ones(*xs.shape[:2]).bool().to(self.device)
        return {
            "xs": xs,
            "conditions": conditions,
            "masks": masks,
            "gt_videos": gt_videos,
        }

    # ---------------------------------------------------------------------
    # Prepare Model, Optimizer, and Metrics
    # ---------------------------------------------------------------------
    def _build_model(self, diffusion_cls: Callable) -> None:
        # 1. Diffusion model
        if self.cfg.compile:
            # NOTE: this compiling is only for training speedup
            if self.cfg.compile == "true_without_ddp_optimizer":
                # NOTE: `cfg.compile` should be set to this value when using `torch.compile` with DDP & Gradient Checkpointing
                # Otherwise, torch.compile will raise an error.
                # Reference: https://github.com/pytorch/pytorch/issues/104674
                # pylint: disable=protected-access
                torch._dynamo.config.optimize_ddp = False
            assert (
                self.cfg.diffusion.is_continuous
            ), "`torch.compile` is only verified for continuous-time diffusion models. To use it for discrete-time models, it should be tested, including # graph breaks"

        self.diffusion_model = torch.compile(
            diffusion_cls(
                cfg=self.cfg.diffusion,
                backbone_cfg=self.cfg.backbone,
                x_shape=self.x_shape,
                max_tokens=self.max_tokens,
                external_cond_type=self.external_cond_type,
                external_cond_num_classes=self.external_cond_num_classes,
                external_cond_dim=self.external_cond_dim,
            ),
            disable=not self.cfg.compile,
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

        # 2. VAE model
        if self.is_latent_diffusion and self.is_latent_online:
            self._load_vae()
        else:
            self.vae = None

        # 3. Metrics
        if len(self.tasks) == 0:
            return

    def get_val_dataloader_name(self, dataloader_idx: int) -> str:
        """
        Get the string representation of the dataloader index.
        """
        val_dataloaders = self.trainer.val_dataloaders
        return list(val_dataloaders.keys())[dataloader_idx]
    
    def check_history_free_validation(self, dataloader_name):
        """
        Check if the validation dataloader is history-free.
        """
        return "history_free" in dataloader_name or "unconditional" in dataloader_name

    
    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0, namespace="validation") -> STEP_OUTPUT:
        """
        dataloader_idx: 0 for training, 1 for validation
        """
        # 1. If running validation while training a model, directly evaluate
        # the denoising performance to detect overfitting, etc.
        # Logs the "denoising_vis" visualization as well as "validation/loss" metric.
        # if self.trainer.state.fn == "FIT":

        # NOTE: after 1 dataloader done, will call `on_validation_dataloader_end` to calculate metrics
        # The last dataloader will be processed in `on_validation_epoch_end`
        if dataloader_idx != self.validation_dataloader_idx:
            if self.validation_dataloader_idx >= 0:
                self.on_validation_dataloader_end()
            self.validation_dataloader_idx = dataloader_idx

        dataloader_name = self.get_val_dataloader_name(dataloader_idx)
        self._eval_denoising(batch, batch_idx, namespace=dataloader_name)

        # 2. Sample all videos (based on the specified tasks)
        # and log the generated videos and metrics.
        if not (
            self.trainer.sanity_checking and not self.cfg.logging.sanity_generation
        ):  
            # sample history-guided videos
            n_context_tokens = 0 if self.check_history_free_validation(dataloader_name) else self.n_context_tokens
            n_context_frames = 0 if self.check_history_free_validation(dataloader_name) else self.n_context_frames

            all_videos = self._sample_all_videos(batch, batch_idx, dataloader_name, n_context_tokens=n_context_tokens)
            self._update_metrics(all_videos)
            self._log_videos(all_videos, dataloader_name, n_context_frames)
        
        if self.cfg.save_attn_map.enabled:
            # TODO: unconditional 
            save_attention_maps(attn_maps, self.cfg.save_attn_map.attn_map_dir, False, batch_idx)
    
    @torch.no_grad()
    def new_validation_step(self, batch, batch_idx, accelerator: Accelerator, namespace="validation") -> STEP_OUTPUT:
        """
        dataloader_idx: 0 for training, 1 for validation
        """
        # 1. If running validation while training a model, directly evaluate
        # the denoising performance to detect overfitting, etc.
        # Logs the "denoising_vis" visualization as well as "validation/loss" metric.
        # if self.trainer.state.fn == "FIT":

        # NOTE: after 1 dataloader done, will call `on_validation_dataloader_end` to calculate metrics
        # The last dataloader will be processed in `on_validation_epoch_end`
        # if dataloader_idx != self.validation_dataloader_idx:
        #     if self.validation_dataloader_idx >= 0:
        #         self.on_validation_dataloader_end()
        #     self.validation_dataloader_idx = dataloader_idx

        # dataloader_name = self.get_val_dataloader_name(dataloader_idx)
        denoising_output = self._new_eval_denoising(batch, batch_idx, namespace=namespace)

        # 2. Sample all videos (based on the specified tasks)
        # and log the generated videos and metrics.
        all_videos = self._sample_all_videos(batch, batch_idx, namespace, n_context_tokens=self.n_context_tokens)
        # self._log_videos(all_videos, namespace, self.n_context_frames)
        
        if self.cfg.save_attn_map.enabled:
            # TODO: unconditional 
            save_attention_maps(attn_maps, self.cfg.save_attn_map.attn_map_dir, False, batch_idx)

        return denoising_output, all_videos


    def _eval_denoising(self, batch, batch_idx, namespace="training") -> None:
        """Evaluate the denoising performance during training."""
        xs = batch["xs"]
        conditions = batch.get("conditions")
        masks = torch.ones(*xs.shape[:2]).bool().to(self.device)
        gt_videos = batch.get("gt_videos")

        xs = xs[:, : self.max_tokens]
        if conditions is not None:
            match self.external_cond_type:
                case "label":
                    conditions = conditions
                case "action":
                    conditions = conditions[:, : self.max_tokens]
                case _:
                    raise ValueError(
                        f"Unknown external condition type: {self.external_cond_type}. "
                        "Supported types are 'label' and 'action'."
                    )
                
        masks = masks[:, : self.max_tokens]
        if gt_videos is not None:
            gt_videos = gt_videos[:, : self.max_frames]

        eval_batch = {
            "xs": xs,
            "conditions": conditions,
            "masks": masks,
            "gt_videos": gt_videos,
        }

        output = self.training_step(eval_batch, batch_idx, namespace=namespace)

        gt_videos = output["xs"]
        recons = output["xs_pred"]
        if self.is_latent_diffusion:
            recons = self._decode(recons)
            gt_videos = self._decode(gt_videos)

        if recons.shape[1] < gt_videos.shape[1]:  # recons.ndim is 5
            recons = F.pad(
                recons,
                (0, 0, 0, 0, 0, 0, 0, gt_videos.shape[1] - recons.shape[1], 0, 0),
            )

        gt_videos, recons = self.gather_data((gt_videos, recons))

        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.logging.max_num_videos
        ):
            return

        num_videos_to_log = min(
            self.logging.max_num_videos - self.num_logged_videos,
            gt_videos.shape[0],
        )
        log_video(
            recons[:num_videos_to_log],
            gt_videos[:num_videos_to_log],
            step=self.global_step,
            namespace=f"{namespace}_denoising_vis",
            logger=self.logger.experiment,
            indent=self.num_logged_videos,
            captions="denoised | gt",
            resize_to=(64,64)
        )
    
    def _new_eval_denoising(self, batch, batch_idx, namespace="training") -> Dict[str, Tensor]:
        """Evaluate the denoising performance during training."""
        xs = batch["xs"]
        conditions = batch.get("conditions")
        masks = torch.ones(*xs.shape[:2]).bool().to(self.device)
        gt_videos = batch.get("gt_videos")

        xs = xs[:, : self.max_tokens]
        if conditions is not None:
            match self.external_cond_type:
                case "label":
                    conditions = conditions
                case "action":
                    conditions = conditions[:, : self.max_tokens]
                case _:
                    raise ValueError(
                        f"Unknown external condition type: {self.external_cond_type}. "
                        "Supported types are 'label' and 'action'."
                    )
                
        masks = masks[:, : self.max_tokens]
        if gt_videos is not None:
            gt_videos = gt_videos[:, : self.max_frames]

        eval_batch = {
            "xs": xs,
            "conditions": conditions,
            "masks": masks,
            "gt_videos": gt_videos,
        }

        output = self.training_step(eval_batch, batch_idx, namespace=namespace)

        gt_videos = output["xs"]
        recons = output["xs_pred"]
        if self.is_latent_diffusion:
            recons = self._decode(recons)
            gt_videos = self._decode(gt_videos)

        if recons.shape[1] < gt_videos.shape[1]:  # recons.ndim is 5
            recons = F.pad(
                recons,
                (0, 0, 0, 0, 0, 0, 0, gt_videos.shape[1] - recons.shape[1], 0, 0),
            )

        output['gts'] = gt_videos
        output['recons'] = recons
        return output

    def on_validation_epoch_start(self) -> None:
        if self.cfg.logging.deterministic is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.global_rank
                + self.trainer.world_size * self.cfg.logging.deterministic
            )
        if self.is_latent_diffusion and not self.is_latent_online:
            self._load_vae()
        
        if self.cfg.save_attn_map.enabled:
            register_hooks(self.diffusion_model.model, True)

        self.val_metrics = {}

        self.validation_dataloader_idx = -1

    def new_on_validation_epoch_start(self) -> None:
        if self.is_latent_diffusion and not self.is_latent_online:
            self._load_vae()
        
        if self.cfg.save_attn_map.enabled:
            register_hooks(self.diffusion_model.model, True)

        self.val_metrics = {}
        self.validation_dataloader_idx = -1

    def on_validation_dataloader_end(self) -> None:
        """
        Called at the end of the validation dataloader.
        This is used to reset the generator and vae for the next epoch.
        """
        namespace = self.get_val_dataloader_name(self.validation_dataloader_idx)
        print('Done with validation dataloader:', namespace)
        if self.trainer.sanity_checking and not self.cfg.logging.sanity_generation:
            return

        for task in self.tasks:
            metrics = self._metrics(task).log(task)
            metrics = {f"{namespace}_{k}": v for k, v in metrics.items()}

            self.val_metrics.update(metrics)
        
        self.validation_dataloader_idx = None
        self.num_logged_videos = 0

    def on_validation_epoch_end(self, namespace="validation") -> None:
        """
        dataloader_idx: 0 for training, 1 for validation
        """
        self.on_validation_dataloader_end()

        self.generator = None
        if self.is_latent_diffusion and not self.is_latent_online:
            self.vae = None

        if self.cfg.save_attn_map:
            clear_hooks(self.diffusion_model.model)

        if self.trainer.sanity_checking and not self.cfg.logging.sanity_generation:
            return

        if 'validation_history_guided_prediction/fvd' in self.val_metrics.keys():
            self.val_metrics['prediction/fvd'] = self.val_metrics.get('validation_history_guided_prediction/fvd')
        elif 'validation_history_free_prediction/fvd' in self.val_metrics.keys():
            self.val_metrics['prediction/fvd'] = self.val_metrics.get('validation_history_free_prediction/fvd')
        else:
            raise ValueError(f"FVD metric not found in {namespace} metrics: {self.val_metrics.keys()}")
        print('self.val_metrics', self.val_metrics)
        # Log the metrics for the validation epoch
        self.log_dict(
                self.val_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                add_dataloader_idx=False
            )
        
        self.val_metrics = None

    def new_on_validation_epoch_end(self, namespace="validation") -> None:
        self.generator = None
        if self.is_latent_diffusion and not self.is_latent_online:
            self.vae = None

        if self.cfg.save_attn_map:
            clear_hooks(self.diffusion_model.model)


    # ---------------------------------------------------------------------
    # Test step
    # ---------------------------------------------------------------------
    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")


    # ---------------------------------------------------------------------
    # Normalization Utils
    # ---------------------------------------------------------------------
    # scale factor
    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    # un-scale factor
    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean

    # ---------------------------------------------------------------------
    # Latent
    # ---------------------------------------------------------------------
    def _load_vae(self) -> None:
        """
        Load the pretrained VAE model.
        """
        if hasattr(self.cfg.vae, 'name') and "dc_ae" in self.cfg.vae.name:
            # NOTE: this is a special case for the DC-AE preprocessor
            # that is used for training the latent diffusion model
            self.vae = MyAutoencoderDC.from_pretrained(
                cfg=self.cfg.vae,
                torch_dtype=(
                    torch.float16 if self.cfg.vae.use_fp16 else torch.float32
                ),
                **self.cfg.vae.pretrained_kwargs
            ).to(self.device)
        elif hasattr(self.cfg.vae, 'name') and self.cfg.vae.name == "kl_autoencoder":
            self.vae = AutoencoderKL.from_pretrained(
                **self.cfg.vae
            ).to(self.device)
        elif hasattr(self.cfg.vae, 'name') and self.cfg.vae.name == "titok_kl_preprocessor":
            self.vae = TiTok_KL(
                image_size=self.cfg.vae.image_size,
                token_size=self.cfg.vae.token_size,
                use_l2_norm=self.cfg.vae.use_l2_norm,
                vit_enc_model_size=self.cfg.vae.vit_enc_model_size,
                vit_dec_model_size=self.cfg.vae.vit_dec_model_size,
                vit_enc_patch_size=self.cfg.vae.vit_enc_patch_size,
                vit_dec_patch_size=self.cfg.vae.vit_dec_patch_size,
                num_latent_tokens=self.cfg.vae.num_latent_tokens,
                use_checkpoint=self.cfg.vae.use_checkpoint,
            )
            import safetensors
            state_dict = safetensors.torch.load_file(self.cfg.vae.pretrained_path, device='cpu')
            self.vae.load_state_dict(state_dict, strict=True)
            self.vae.to(self.device)
        else:
            vae_cls = VideoVAE if self.is_latent_video_vae else ImageVAE
            self.vae = vae_cls.from_pretrained(
                path=self.cfg.vae.pretrained_path,
                torch_dtype=(
                    torch.float16 if self.cfg.vae.use_fp16 else torch.float32
                ),  # only for Diffuser's ImageVAE
                **self.cfg.vae.pretrained_kwargs,
            ).to(self.device)
            
        freeze_model(self.vae)

    @torch.no_grad()
    def _run_vae(
        self,
        x: Tensor,
        shape: str,
        vae_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """
        Helper function to run the VAE, either for encoding or decoding.
        - Requires shape to be a permutation of b, t, c, h, w.
        - Reshapes the input tensor to the required shape for the VAE, and reshapes the output back.
            - x: `shape` shape.
            - VideoVAE requires (b, c, t, h, w) shape.
            - ImageVAE requires (b, c, h, w) shape.
        - Split the input tensor into chunks of size cfg.vae.batch_size, to avoid memory errors.
        """
        x = rearrange(x, f"{shape} -> b c t h w")
        batch_size = x.shape[0]
        vae_batch_size = self.cfg.vae.batch_size
        # chunk the input tensor by vae_batch_size
        chunks = torch.chunk(x, (batch_size + vae_batch_size - 1) // vae_batch_size, 0)
        outputs = []
        for chunk in chunks:
            b = chunk.shape[0]
            if not self.is_latent_video_vae:
                chunk = rearrange(chunk, "b c t h w -> (b t) c h w")

            output = vae_fn(chunk)

            if not self.is_latent_video_vae:
                output = rearrange(output, "(b t) c h w -> b c t h w", b=b)
            outputs.append(output)
        return rearrange(torch.cat(outputs, 0), f"b c t h w -> {shape}")
        
    def _encode(self, x: Tensor, shape: str = "b t c h w") -> Tensor:
        if isinstance(self.vae, MyAutoencoderDC):
            fn = lambda y: self.vae.encode(2.0 * y - 1.0)
        elif isinstance(self.vae, AutoencoderKL):
            fn = lambda y: self.vae.encode(2.0 * y - 1.0, return_dict=False)[0].sample()
        elif isinstance(self.vae, TiTok_KL):
            fn = lambda y: self.vae.encode(y, sample=True)
        else:
            fn = lambda y: self.vae.encode(2.0 * y - 1.0).sample()
        return self._run_vae(x, shape, fn)

    def _decode(self, latents: Tensor, shape: str = "b t c h w") -> Tensor:
        if isinstance(self.vae, MyAutoencoderDC):
            return self._run_vae(
                latents,
                shape,
                lambda y: self.vae.decode(y) * 0.5 + 0.5
            )
        elif isinstance(self.vae, AutoencoderKL):
            return self._run_vae(
                latents,
                shape,
                lambda y: self.vae.decode(y, return_dict=False)[0] * 0.5 + 0.5
            )
        elif isinstance(self.vae, TiTok_KL):
            # Output [0,1]
            return self._run_vae(
                latents,
                shape,
                lambda y: self.vae.decode(y)
            )
        else:
            return self._run_vae(
                latents,
                shape,
                lambda y: (
                    self.vae.decode(y, self._n_tokens_to_n_frames(latents.shape[1]))
                    if self.is_latent_video_vae
                    else self.vae.decode(y)
                )
                * 0.5
                + 0.5,
            )
    
    # ---------------------------------------------------------------------
    # Data Preprocessing Utils
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _process_conditions(
        self,
        conditions: Optional[Tensor],
        noise_levels: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """
        Post-process the conditions before feeding them to the model.
        For example, conditions that should be computed relatively (e.g. relative poses)
        should be processed here instead of the dataset.

        Args:
            conditions (Optional[Tensor], "B T ..."): The external conditions for the video.
            noise_levels (Optional[Tensor], "B T"): Current noise levels for each token during sampling
        """

        if conditions is None:
            return conditions
        match self.cfg.external_cond_processing:
            case None:
                return conditions
            case "mask_first":
                mask = torch.ones_like(conditions)
                # Remove the first token from the conditions
                mask[:, :1, : self.external_cond_dim] = 0
                return conditions * mask
            case _:
                raise NotImplementedError(
                    f"External condition processing {self.cfg.external_cond_processing} is not implemented."
                )
            
    def _pad_to_max_tokens(self, y: Optional[Tensor]) -> Tensor:
        """Given a tensor y of shape (B, T, ...), pad it at the end across the time dimension to have a length of self.max_tokens."""
        if y is None:
            return y
        if y.shape[1] < self.max_tokens:
            y = torch.cat(
                [
                    y,
                    repeat(
                        y[:, -1:],
                        "b 1 ... -> b t ...",
                        t=self.max_tokens - y.shape[1],
                    ),
                ],
                dim=1,
            )
        return y
    
    def _reweight_loss(self, loss, weight=None):
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape)
            weight = rearrange(
                weight,
                "... -> ..." + " 1" * expand_dim,
            )
            loss = loss * weight

        return loss.mean()

    # ---------------------------------------------------------------------
    # Tensor utils
    # ---------------------------------------------------------------------   
    def _extend_x_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Extend the tensor by adding dimensions at the end to match x_stacked_shape."""
        return rearrange(x, "... -> ..." + " 1" * len(self.x_shape))
    
    # ---------------------------------------------------------------------
    # Logging (Metrics, Videos)
    # ---------------------------------------------------------------------

    def _metrics(
        self,
        task: Literal["prediction", "interpolation"],
    ) -> Optional[VideoMetric]:
        """
        Get the appropriate metrics object for the given task.
        """
        return getattr(self, f"metrics_{task}", None)
    
    def _update_metrics(self, all_videos: Dict[str, Tensor]) -> None:
        """Update all metrics during validation/test step."""
        if (
            self.logging.n_metrics_frames is not None
        ):  # only consider the first n_metrics_frames for evaluation
            all_videos = {
                k: v[:, : self.logging.n_metrics_frames] for k, v in all_videos.items()
            }

        gt_videos = all_videos["gt"]
        for task in self.tasks:
            metric = self._metrics(task)
            videos = all_videos[task]
            context_mask = torch.zeros(videos.shape[1]).bool().to(self.device)
            match task:
                case "prediction":
                    context_mask[: self.n_context_frames] = True
                case "interpolation":
                    context_mask[[0, -1]] = True
            if self.logging.n_metrics_frames is not None:
                context_mask = context_mask[: self.logging.n_metrics_frames]
            metric(videos, gt_videos, context_mask=context_mask)
    
    def _log_videos(self, all_videos: Dict[str, Tensor], namespace: str, context_frames=None) -> None:
        """Log videos during validation/test step."""
        all_videos = self.gather_data(all_videos)
        batch_size, n_frames = all_videos["gt"].shape[:2]
        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.logging.max_num_videos
        ):
            return

        num_videos_to_log = min(
            self.logging.max_num_videos - self.num_logged_videos,
            batch_size,
        )
        cut_videos = lambda x: x[:num_videos_to_log]

        if context_frames is None:
            context_frames=self.n_context_frames if task == "prediction" else torch.tensor([0, n_frames - 1], device=self.device, dtype=torch.long)


        for task in self.tasks:
            # (f"{namespace}_{task}_vis")
            log_video(
                cut_videos(all_videos[task]),
                cut_videos(all_videos["gt"]),
                step=None if namespace == "test" else self.global_step,
                namespace=f"{namespace}_{task}_vis",
                logger=self.logger.experiment,
                indent=self.num_logged_videos,
                raw_dir=self.logging.raw_dir,
                context_frames=context_frames,
                captions=f"{task} | gt",
                resize_to=(64,64)
            )

        self.num_logged_videos += batch_size

    #----------------------------------------------------------------------
    # Noise scheduling
    #----------------------------------------------------------------------
    def _get_training_noise_levels(
        self, xs: Tensor, masks: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """Generate random noise levels for training."""
        batch_size, n_tokens, *_ = xs.shape

        # random function different for continuous and discrete diffusion
        rand_fn = partial(
            *(
                (torch.rand,)
                if self.cfg.diffusion.is_continuous
                else (torch.randint, 0, self.timesteps)
            ),
            device=xs.device,
            generator=self.generator,
        )

        # baseline training (SD: fixed_context, BD: variable_context)
        context_mask = None
        if self.cfg.variable_context.enabled:
            assert (
                not self.cfg.fixed_context.enabled
            ), "Cannot use both fixed and variable context"
            context_mask = bernoulli_tensor(
                (batch_size, n_tokens),
                self.cfg.variable_context.prob,
                device=self.device,
                generator=self.generator,
            ).bool()
        elif self.cfg.fixed_context.enabled:
            context_indices = self.cfg.fixed_context.indices or list(
                range(self.n_context_tokens)
            )
            context_mask = torch.zeros(
                (batch_size, n_tokens), dtype=torch.bool, device=xs.device
            )
            context_mask[:, context_indices] = True

        match self.cfg.noise_level:
            case "random_independent":  # independent noise levels (Diffusion Forcing)
                noise_levels = rand_fn((batch_size, n_tokens))
            case "random_uniform":  # uniform noise levels (Typical Video Diffusion)
                noise_levels = rand_fn((batch_size, 1)).repeat(1, n_tokens)
            case "interleaved":
                odd_noise_level = rand_fn((batch_size, 1))
                even_noise_level = rand_fn((batch_size, 1))
                noise_levels = torch.zeros(batch_size, n_tokens, device=xs.device, dtype=odd_noise_level.dtype)
                noise_levels[:, ::2] = odd_noise_level
                noise_levels[:, 1::2] = even_noise_level
                

        if self.cfg.uniform_future.enabled:  # simplified training (Appendix A.5)
            noise_levels[:, self.n_context_tokens :] = rand_fn((batch_size, 1)).repeat(
                1, n_tokens - self.n_context_tokens
            )

        # treat frames that are not available as "full noise"
        noise_levels = torch.where(
            reduce(masks.bool(), "b t ... -> b t", torch.any),
            noise_levels,
            torch.full_like(
                noise_levels,
                1 if self.cfg.diffusion.is_continuous else self.timesteps - 1,
            ),
        )

        if context_mask is not None:
            # binary dropout training to enable guidance
            dropout = (
                (
                    self.cfg.variable_context
                    if self.cfg.variable_context.enabled
                    else self.cfg.fixed_context
                ).dropout
                if self.trainer.training
                else 0.0
            )
            context_noise_levels = bernoulli_tensor(
                (batch_size, 1),
                dropout,
                device=xs.device,
                generator=self.generator,
            )
            if not self.cfg.diffusion.is_continuous:
                context_noise_levels = context_noise_levels.long() * (
                    self.timesteps - 1
                )
            noise_levels = torch.where(context_mask, context_noise_levels, noise_levels)

            # modify masks to exclude context frames from loss computation
            context_mask = rearrange(
                context_mask, "b t -> b t" + " 1" * len(masks.shape[2:])
            )
            masks = torch.where(context_mask, False, masks)

        return noise_levels, masks
    

    def _generate_scheduling_matrix(
        self,
        horizon: int,
        padding: int = 0,
    ):
        match self.cfg.scheduling_matrix:
            case "full_sequence" | "gibbs":
                scheduling_matrix = np.arange(self.sampling_timesteps, -1, -1)[
                    :, None
                ].repeat(horizon, axis=1)
            case "autoregressive":
                scheduling_matrix = self._generate_pyramid_scheduling_matrix(
                    horizon, self.sampling_timesteps
                )
            case "interleaved":
                scheduling_matrix = self._generate_interleaved_scheduling_matrix(
                    horizon, 3, self.sampling_timesteps
                )

        scheduling_matrix = torch.from_numpy(scheduling_matrix).long()
        scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(
            scheduling_matrix
        )

        if self.cfg.scheduling_matrix == "gibbs":
            n_sampling_steps = scheduling_matrix.shape[0]
            scheduling_matrix = repeat(scheduling_matrix, 't b -> (t h) b', h=horizon)
            for i in range(1, n_sampling_steps):
                for j in range(horizon):
                    scheduling_matrix[i * horizon + j, j+1:] = scheduling_matrix[(i-1) * horizon + horizon - 1, j+1:]

        # paded entries are labeled as pure noise
        scheduling_matrix = F.pad(
            scheduling_matrix, (0, padding, 0, 0), value=self.timesteps - 1
        )
        
        return scheduling_matrix

    def _generate_interleaved_scheduling_matrix(
        self,
        horizon: int,
        interleaved_size = 2,
        sampling_timesteps: int = 50,
    ):
        noise_levels = []
        max_length = sampling_timesteps + interleaved_size
        for i in range(horizon):
            # only for non-symmetric case
            start_idx = i%interleaved_size + 1
            # start_idx = interleaved_size - i%interleaved_size
            cur_noise_levels = [sampling_timesteps]*start_idx
            for j in range(sampling_timesteps):
                noise_idx = max(sampling_timesteps - start_idx - interleaved_size*j, 0)
                if noise_idx == 0:
                    cur_noise_levels += [noise_idx] * (max_length - len(cur_noise_levels))
                    break
                else:
                    cur_noise_levels += [noise_idx] * interleaved_size
                    
            noise_levels.append(cur_noise_levels)
        noise_levels = np.array(noise_levels)
        return noise_levels.T

    def _generate_pyramid_scheduling_matrix(self, horizon: int, sampling_timesteps:int, uncertainty_scale: float = 1.0):
        height = sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, sampling_timesteps)

    def _generate_refine_scheduling_matrix(
        self,
        horizon: int,
        goback_length: int,
        n_goback: int,
        padding: int = 0,
    ):
        assert self.cfg.scheduling_matrix == 'full_sequence', "Refining only support full_sequence scheduling matrix"
        scheduling_matrix = np.arange(self.sampling_timesteps, -1, -1)
        final_scheduling_matrix = []

        goback_idxs = [i for i in range(1, self.sampling_timesteps - goback_length, goback_length)]
        for t in scheduling_matrix:
            final_scheduling_matrix.append(t)
            if t in goback_idxs:
                for i in range(n_goback):
                    final_scheduling_matrix = final_scheduling_matrix + [s for s in range(t+1,t+goback_length+1)]
                    final_scheduling_matrix = final_scheduling_matrix + [s for s in range(t+goback_length-1,t-1,-1)]
        
        final_scheduling_matrix = torch.tensor(final_scheduling_matrix).long()
        scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(final_scheduling_matrix)[:, None].repeat(1, horizon)
        
        # paded entries are labeled as pure noise
        scheduling_matrix = F.pad(
            scheduling_matrix, (0, padding, 0, 0), value=self.timesteps - 1
        )

        return scheduling_matrix

    # ---------------------------------------------------------------------
    # Length-related Properties and Utils
    # NOTE: "Frame" and "Token" should be distinguished carefully.
    # "Frame" refers to original unit of data loaded from dataset.
    # "Token" refers to the unit of data processed by the diffusion model.
    # The two differ when using a VAE for latent diffusion.
    # ---------------------------------------------------------------------

    def _n_frames_to_n_tokens(self, n_frames: int) -> int:
        """
        Converts the number of frames to the number of tokens.
        - Chunk-wise VideoVAE: 1st frame -> 1st token, then every self.temporal_downsampling_factor frames -> next token.
        - ImageVAE or Non-latent Diffusion: 1 token per frame.
        """
        return (n_frames - 1) // self.temporal_downsampling_factor + 1

    def _n_tokens_to_n_frames(self, n_tokens: int) -> int:
        """
        Converts the number of tokens to the number of frames.
        """
        return (n_tokens - 1) * self.temporal_downsampling_factor + 1

    # ---------------------------------------------------------------------
    # NOTE: max_{frames, tokens} indicates the maximum number of frames/tokens
    # that the model can process within a single forward pass.
    # ---------------------------------------------------------------------

    @property
    def max_frames(self) -> int:
        return self.cfg.max_frames

    @property
    def max_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.max_frames)

    # ---------------------------------------------------------------------
    # NOTE: n_{frames, tokens} indicates the number of frames/tokens
    # that the model actually processes during training/validation.
    # During validation, it may be different from max_{frames, tokens},
    # ---------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return self.max_frames if self.trainer.training else self.cfg.n_frames

    @property
    def n_context_frames(self) -> int:
        return self.cfg.context_frames

    @property
    def n_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.n_frames)

    @property
    def n_context_tokens(self) -> int:
        return self._n_frames_to_n_tokens(self.n_context_frames)
    
    # ---------------------------------------------------------------------
    # Optimizers and Schedulers
    # ---------------------------------------------------------------------
    def get_params(self):
        """
        Returns the parameters to optimize.
        This is used to filter out the parameters that are not trainable.
        """
        params = {
            'frozen': [],
            'trainable': [],
            'total_frozen': 0,
            'total_trainable': 0
        }

        for name, param in self.diffusion_model.named_parameters():
            if param.requires_grad:
                params['trainable'].append(name)
                params['total_trainable'] += param.numel()
            else:
                params['frozen'].append(name)
                params['total_frozen'] += param.numel()
            
        params['total'] = params['total_frozen'] + params['total_trainable']
        return params
    
    def configure_optimizers(self):
        transition_params = list(self.diffusion_model.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            transition_params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )

        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer_dynamics,
                **self.cfg.lr_scheduler,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer_dynamics,
            "lr_scheduler": lr_scheduler_config,
        }
    
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if (
            self.cfg.logging.grad_norm_freq
            and self.global_step % self.cfg.logging.grad_norm_freq == 0
        ):
            norms = grad_norm(self.diffusion_model, norm_type=2)
            # NOTE: `norms` need not be gathered, as they are already uniform across all devices
            self.log_dict(norms)

    # ---------------------------------------------------------------------
    # Checkpoint Utils
    # ---------------------------------------------------------------------
    def _uncompile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict if self.diffusion_model is compiled, to uncompiled."""
        if self.cfg.compile:
            checkpoint["state_dict"] = {
                k.replace("diffusion_model._orig_mod.", "diffusion_model."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _compile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict to the format expected by the compiled model."""
        if self.cfg.compile:
            checkpoint["state_dict"] = {
                k.replace("diffusion_model.", "diffusion_model._orig_mod."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _should_include_in_checkpoint(self, key: str) -> bool:
        return key.startswith("diffusion_model.model") or key.startswith(
            "diffusion_model._orig_mod.model"
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) uncompile the model's state_dict before saving
        self._uncompile_checkpoint(checkpoint)
        # 2. Only save the meaningful keys defined by self._should_include_in_checkpoint
        # by default, only the model's state_dict is saved and metrics & registered buffes (e.g. diffusion schedule) are not discarded
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if not self._should_include_in_checkpoint(key):
                del state_dict[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) compile the model's state_dict before loading
        self._compile_checkpoint(checkpoint)
        # 2. (Optionally) swap the state_dict of the model with the EMA weights for inference
        super().on_load_checkpoint(checkpoint)
        # 3. (Optionally) reset the optimizer states - for fresh finetuning or resuming training
        if self.cfg.checkpoint.reset_optimizer:
            checkpoint["optimizer_states"] = []

        # 4. Rewrite the state_dict of the checkpoint, only leaving meaningful keys
        # defined by self._should_include_in_checkpoint
        # also print out warnings when the checkpoint does not exactly match the expected format

        new_state_dict = {}
        for key, value in self.state_dict().items():
            if (
                self._should_include_in_checkpoint(key)
                and key in checkpoint["state_dict"]
            ):
                new_state_dict[key] = checkpoint["state_dict"][key]
            else:
                new_state_dict[key] = value

        # print keys that are ignored from the checkpoint
        ignored_keys = [
            key
            for key in checkpoint["state_dict"].keys()
            if not self._should_include_in_checkpoint(key)
        ]
        if ignored_keys:
            rank_zero_print(
                cyan("The following keys are ignored from the checkpoint:"),
                ignored_keys,
            )
        # print keys that are not found in the checkpoint
        missing_keys = [
            key
            for key in self.state_dict().keys()
            if self._should_include_in_checkpoint(key)
            and key not in checkpoint["state_dict"]
        ]
        if missing_keys:
            rank_zero_print(
                cyan("The following keys are not found in the checkpoint:"),
                missing_keys,
            )
            if self.cfg.checkpoint.strict:
                raise ValueError(
                    "Thus, the checkpoint cannot be loaded. To ignore this error, turn off strict checkpoint loading by setting `algorithm.checkpoint.strict=False`."
                )
            else:
                rank_zero_print(
                    cyan(
                        "Strict checkpoint loading is turned off, so using the initialized value for the missing keys."
                    )
                )
        checkpoint["state_dict"] = new_state_dict

    def _load_ema_weights_to_state_dict(self, checkpoint: Dict[str, Any]) -> None:
        if (
            checkpoint.get("pretrained_ema", False)
            and len(checkpoint["optimizer_states"]) == 0
        ):
            # NOTE: for lightweight EMA-only ckpts for releasing pretrained models,
            # we already have EMA weights in the state_dict
            return
        ema_weights = checkpoint["optimizer_states"][0]["ema"]
        parameter_keys = [
            "diffusion_model." + k for k, _ in self.diffusion_model.named_parameters()
        ]
        assert len(parameter_keys) == len(
            ema_weights
        ), "Number of original weights and EMA weights do not match."
        for key, weight in zip(parameter_keys, ema_weights):
            checkpoint["state_dict"][key] = weight

    # ---------------------------------------------------------------------
    # Config Utils
    # ---------------------------------------------------------------------

    def check_video_vae_compatibility(self, cfg: DictConfig):
        """
        Check if the configuration is compatible with VideoVAE.
        Currently, it is not compatible with many functionalities, due to complicated shape/length changes.
        """
        assert (
            cfg.latent.type == "online"
        ), "Latents must be processed online when using VideoVAE."
        assert (
            cfg.external_cond_dim == 0
        ), "External conditions are not supported yet when using VideoVAE."
