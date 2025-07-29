from typing import Optional, Dict, Callable, Tuple
from functools import partial
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from lightning.pytorch.utilities.types import STEP_OUTPUT
from einops import rearrange, repeat, reduce
from tqdm import tqdm
from accelerate import Accelerator
from algorithms.common.base_pytorch_video_algo import BaseVideoAlgo
from utils.distributed_utils import is_rank_zero
from utils.torch_utils import bernoulli_tensor
from utils.logging_utils import log_video
from .diffusion import (
    DiscreteDiffusion,
    ContinuousDiffusion,
)
from .history_guidance import HistoryGuidance

torch.set_printoptions(linewidth=100000, threshold=10000)
class DifferenceDFoTVideo(BaseVideoAlgo):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) for video generation.
    """
    def __init__(self, cfg: DictConfig):
        assert cfg.backbone.merge_type in ["concat", "interleaved"], f"Unsupported merge type: {cfg.backbone.merge_type}"
        super().__init__(cfg)
        self.merge_type = cfg.backbone.merge_type

    def _build_model(self) -> None:
        diffusion_cls = (
            ContinuousDiffusion
            if self.cfg.diffusion.is_continuous
            else DiscreteDiffusion
        )
        super()._build_model(diffusion_cls=diffusion_cls)

    @property
    def diff_max_tokens(self) -> int:
        return self.max_tokens * 2
    
    @torch.no_grad()
    def merge_tensors(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Merge two tensors based on the merge type.
        """ 
        if x is None or y is None:
            return None
        assert x.shape == y.shape, "Tensors must have the same shape to be merged."
        if self.merge_type == 'concat':
            return torch.cat([x, y], dim=1)
        elif self.merge_type == 'interleaved':
            return rearrange(
                torch.stack([x, y], dim=-1),
                "b t ... two -> b (t two) ...",
            )
        else:
            raise ValueError(f"Unsupported merge type: {self.merge_type}")
    
    @torch.no_grad()
    def unmerge_tensors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unmerge the merged tensor based on the merge type.
        """
        if self.merge_type == 'concat':
            return x.chunk(2, dim=1)
        elif self.merge_type == 'interleaved':
            return rearrange(
                x, "b (t two) ... -> (two b) t ...", two=2
            ).chunk(2, dim=0)
        else:
            raise ValueError(f"Unsupported merge type: {self.merge_type}")        

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        """Training step"""
        xs = batch["xs"] # shape: (B, T*2, C, H, W)
        difference = torch.diff(xs, dim=1, prepend=xs[:, :1])

        conditions = batch.get("conditions")
        masks = batch["masks"]

        # TODO: check if we should use the same noise for xs and difference xs
        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        xs = self.merge_tensors(difference, xs) #TODO: difference should be first
        noise_levels = self.merge_tensors(noise_levels, noise_levels) # Double the noise levels for difference
        org_masks = masks.clone()
        masks = self.merge_tensors(masks, masks)

        conditions = self._process_conditions(conditions)
        conditions = self.merge_tensors(conditions, conditions) if conditions is not None else None
        xs_pred, loss = self.diffusion_model(
            xs,
            conditions,
            k=noise_levels,
        )
        diff_loss, xs_loss = self.unmerge_tensors(loss)
        loss = self._reweight_loss(loss, masks)
        diff_loss = self._reweight_loss(diff_loss.detach(), org_masks)
        xs_loss = self._reweight_loss(xs_loss.detach(), org_masks)

        # if attached to trainer => log
        try:
            if batch_idx % self.cfg.logging.loss_freq == 0:
                self.log(
                    f"{namespace}/loss",
                    loss,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                    add_dataloader_idx=False
                )
                self.log(
                    f"{namespace}/diff_loss",
                    diff_loss,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                    add_dataloader_idx=False
                )
                self.log(
                    f"{namespace}/xs_loss",
                    xs_loss,
                    on_step=namespace == "training",
                    on_epoch=namespace != "training",
                    sync_dist=True,
                    add_dataloader_idx=False
                )
        except AttributeError:
            pass

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))
        output_dict = {
            "loss": loss,
            'diff_loss': diff_loss,
            'xs_loss': xs_loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }
        return output_dict

    @torch.no_grad()
    def new_validation_step(self, batch, batch_idx, accelerator: Accelerator, namespace="validation") -> STEP_OUTPUT:
        """
        dataloader_idx: 0 for training, 1 for validation
        """
        # dataloader_name = self.get_val_dataloader_name(dataloader_idx)
        denoising_output = self._new_eval_denoising(batch, batch_idx, namespace=namespace)
        # 2. Sample all videos (based on the specified tasks)
        # and log the generated videos and metrics.
        all_videos = self._sample_all_videos(batch, batch_idx, namespace, n_context_tokens=self.n_context_tokens)
        # self._log_videos(all_videos, namespace, self.n_context_frames)
        
        # if self.cfg.save_attn_map.enabled:
        #     # TODO: unconditional 
        #     save_attention_maps(attn_maps, self.cfg.save_attn_map.attn_map_dir, False, batch_idx)

        # return two outputs: denoising output and all_videos
        if accelerator.is_main_process:
            denoising_output = accelerator.gather_for_metrics(denoising_output)
            all_videos = accelerator.gather_for_metrics(all_videos)
            return denoising_output, all_videos
        else:
            return None, None

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
    def _sample_all_videos(
        self, batch, batch_idx, namespace="validation", n_context_tokens=None
    ) -> Optional[Dict[str, Tensor]]:
        gt_videos = batch["gt_videos"]
        xs = batch["xs"]

        all_videos: Dict[str, Tensor] = {"gt": xs.clone()}

        difference = torch.diff(xs, dim=1, prepend=xs[:, :1])
        xs = self.merge_tensors(difference, xs) #TODO: difference should be first

        conditions = batch.get("conditions")
        conditions = self._process_conditions(conditions)
        conditions = self.merge_tensors(conditions, conditions) if conditions is not None else None        

        n_context_tokens = n_context_tokens if n_context_tokens is not None else self.n_context_tokens
        n_context_tokens = n_context_tokens*2 # double the context tokens for difference
        if n_context_tokens > 0:
            assert self.merge_type == "interleaved", "n_context_tokens > 0 is only supported for interleaved merge type"

        for task in self.tasks:
            assert task == 'prediction', "Only prediction task is supported now for DifferenceDFoTVideo"
            sample_fn = (
                self._predict_videos
                if task == "prediction"
                else self._interpolate_videos
            )
            all_videos[task] = sample_fn(xs, conditions=conditions, n_context_tokens=n_context_tokens)
            gen_diff, all_videos[task] = self.unmerge_tensors(all_videos[task]) # unmerge the videos
            all_videos[task + "_diff"] = gen_diff

        # remove None values
        all_videos = {k: v for k, v in all_videos.items() if v is not None}
        # rearrange/unnormalize/detach the videos
        all_videos = {k: self._unnormalize_x(v).detach() for k, v in all_videos.items()}
        # decode latents if using latents
        if self.is_latent_diffusion:
            if gt_videos is None:
                gt_videos = self._decode(all_videos['gt'])

            all_videos = {
                k: self._decode(v) if k != "gt" else gt_videos
                for k, v in all_videos.items()
            }
            all_videos['gt_diff'] = torch.diff(gt_videos, dim=1, prepend=gt_videos[:, :1])
        
        return all_videos

    def _predict_videos(
        self, xs: Tensor, n_context_tokens: int, conditions: Optional[Tensor] = None
    ) -> Tensor:
        """
        Predict the videos with the given context, using sliding window rollouts if necessary.
        Optionally, if cfg.tasks.prediction.keyframe_density < 1, predict the keyframes first,
        then interpolate the missing intermediate frames.
        """
        xs_pred = xs.clone()

        history_guidance = HistoryGuidance.from_config(
            config=self.cfg.tasks.prediction.history_guidance,
            timesteps=self.timesteps,
        )

        density = self.cfg.tasks.prediction.keyframe_density or 1
        if density > 1:
            raise ValueError("tasks.prediction.keyframe_density must be <= 1")
        keyframe_indices = (
            torch.linspace(0, xs_pred.shape[1] - 1, round(density * xs_pred.shape[1]))
            .round()
            .long()
        )
        keyframe_indices = torch.cat(
            [torch.arange(n_context_tokens), keyframe_indices]
        ).unique() # context frames are always keyframes

        if conditions is not None:
            match self.external_cond_type:
                case "label":
                    key_conditions = conditions
                case "action":
                    key_conditions = conditions[:, keyframe_indices]
                case _:
                    raise ValueError(
                        f"Unknown external condition type: {self.external_cond_type}. "
                        "Supported types are 'label' and 'action'."
                    )
        else:
            key_conditions = None
        
        # 1. Predict the keyframes
        xs_pred_key, *_ = self._predict_sequence(
            xs_pred[:, : n_context_tokens],
            length=len(keyframe_indices),
            conditions=key_conditions,
            history_guidance=history_guidance,
            reconstruction_guidance=self.cfg.diffusion.reconstruction_guidance,
            sliding_context_len=self.cfg.tasks.prediction.sliding_context_len or self.max_tokens, # NOTE: Not divided by 2 because we have difference 
        )
        xs_pred[:, keyframe_indices] = xs_pred_key.to(xs_pred.dtype)
        # if is_rank_zero: # uncomment to visualize history guidance
        #     history_guidance.log(logger=self.logger)

        # 2. (Optional) Interpolate the intermediate frames
        if len(keyframe_indices) < xs_pred.shape[1]:
            context_mask = torch.zeros(xs_pred.shape[:2], device=self.device).bool()
            context_mask[:, keyframe_indices] = True
            xs_pred = self._interpolate_videos(
                context=xs_pred,
                context_mask=context_mask,
                conditions=conditions,
            )

        return xs_pred

    def _interpolate_videos(
        self,
        context: Tensor,
        context_mask: Optional[Tensor] = None,
        conditions: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        A general method for frame interpolation. Given a video of any length > 2, when the left and right key frames are known, it (iteratively, if necessary) interpolates the video, filling out all missing frames.

        The logic is as follows:
        1. If the distance between adjacent key frames >= self.max_tokens - 1, it will first infer equally spaced self.max_tokens - 2 frames between the key frames.
        2. Otherwise, it will increase the number of key frames until right before the distance between adjacent key frames > self.max_tokens - 1, then pad the video with with the last key frame (to keep the model input size self.max_tokens).
        3. Repeat the above process until all missing frames are filled.

        Args:
            context (Tensor, "B T C H W"): The video including the context frames.
            context_mask (Optional[Tensor], "B T"): The mask for the context frames. True for the context frames, False otherwise. If None, *only the first and last frames* are considered as key frames. It is assumed that context_mask is identical for all videos in the batch, as "interpolation plan" depends on the context_mask.
            conditions (Optional[Tensor], "B T ..."): The external conditions for the video.
            history_guidance (Optional[HistoryGuidance]): The history guidance object - if None, it will be initialized from the config.
        """
        raise NotImplementedError(
            "Current DifferenceDFoT does not support interpolation. Use the prediction task instead."
        )
        # Generate default context mask if not provided
        if context_mask is None:
            context_mask = torch.zeros(
                context.shape[0], context.shape[1], device=self.device
            ).bool()
            context_mask[:, [0, -1]] = True
        else:
            assert context_mask[
                :, [0, -1]
            ].all(), "The first and last frames must be known to interpolate."

        # enable using different history guidance scheme for interpolation
        history_guidance = HistoryGuidance.from_config(
            config=self.cfg.tasks.interpolation.history_guidance,
            timesteps=self.timesteps,
        )

        # Generate a plan for frame interpolation
        plan = []
        plan_mask = context_mask[0].clone()
        while not plan_mask.all():
            key_frames = torch.where(plan_mask)[0]
            current_plan = []  # plan for the current iteration
            current_chunk = None  # chunk to be merged with the next chunk
            for left, right in zip(key_frames[:-1], key_frames[1:]):
                if current_chunk is not None:
                    if (
                        len(current_chunk) + right - left <= self.max_tokens
                    ):  # merge with the next chunk if possible
                        current_chunk = torch.cat(
                            [
                                current_chunk,
                                torch.arange(
                                    left + 1,
                                    right + 1,
                                    device=self.device,
                                ),
                            ]
                        )
                        continue
                    # if cannot merge, add the current chunk to the plan
                    current_plan.append(current_chunk)
                    current_chunk = None

                if right - left == 1:  # no missing frames
                    continue

                if right - left >= self.max_tokens - 1:  # Case 1
                    current_plan.append(
                        torch.linspace(left, right, self.max_tokens, device=self.device)
                        .round()
                        .long()
                    )
                else:  # Case 2
                    current_chunk = torch.arange(left, right + 1, device=self.device)
            if current_chunk is not None:
                current_plan.append(current_chunk)
            for frames in current_plan:
                plan_mask[frames] = True
            plan.append(current_plan)

        # Execute the plan
        xs = context.clone()
        context_mask = context_mask.clone()
        max_batch_size = self.cfg.tasks.interpolation.max_batch_size
        pbar = tqdm(
            total=sum(
                [
                    (
                        (len(frames) + max_batch_size - 1) // max_batch_size
                        if max_batch_size
                        else 1
                    )
                    for frames in plan
                ]
            )
            * self.sampling_timesteps,
            initial=0,
            desc="Interpolating with DFoT",
            leave=False,
        )
        for current_plan in plan:
            # Collect the batched input for the current plan
            current_context = []
            current_context_mask = []
            current_conditions = [] if conditions is not None else None
            for frames in current_plan:
                current_context.append(self._pad_to_max_tokens(xs[:, frames]))
                current_context_mask.append(
                    self._pad_to_max_tokens(context_mask[:, frames])
                )
                if conditions is not None:
                    match self.external_cond_type:
                        case "label":
                            current_conditions.append(conditions)
                        case "action":
                            current_conditions.append(
                                self._pad_to_max_tokens(conditions[:, frames])
                            )
                        case _:
                            raise ValueError(
                                f"Unknown external condition type: {self.external_cond_type}. "
                                "Supported types are 'label' and 'action'."
                            )

            current_context, current_context_mask, current_conditions = map(
                lambda y: torch.cat(y, 0) if y is not None else None,
                (current_context, current_context_mask, current_conditions),
            )
            xs_pred = []
            # Interpolate the video in parallel,
            # while keeping the batch size smaller than the maximum batch size to avoid memory errors
            max_batch_size = (
                self.cfg.tasks.interpolation.max_batch_size or current_context.shape[0]
            )
            for (
                current_context_chunk,
                current_context_mask_chunk,
                current_conditions_chunk,
            ) in zip(
                current_context.split(max_batch_size, 0),
                current_context_mask.split(max_batch_size, 0),
                (
                    current_conditions.split(max_batch_size, 0)
                    if current_conditions is not None
                    else [None] * (current_context.shape[0] // max_batch_size)
                ),
            ):
                batch_size = current_context_chunk.shape[0]
                if self.cfg.refinement_sampling.enabled:
                    xs_pred_chunk, _ = self._sample_sequence_refine(
                        batch_size=batch_size,
                        context=current_context_chunk,
                        goback_length=self.cfg.refinement_sampling.goback_length,
                        n_goback=self.cfg.refinement_sampling.n_goback,
                        context_mask=current_context_mask_chunk.long(),
                        conditions=current_conditions_chunk,
                        history_guidance=history_guidance,
                        pbar=pbar,
                    )
                else:
                    xs_pred_chunk, _ = self._sample_sequence(
                        batch_size=batch_size,
                        context=current_context_chunk,
                        context_mask=current_context_mask_chunk.long(),
                        conditions=current_conditions_chunk,
                        history_guidance=history_guidance,
                        pbar=pbar,
                    )
                xs_pred.append(xs_pred_chunk)

            xs_pred = torch.cat(xs_pred, 0)
            # Update with the interpolated frames
            for frames, pred in zip(current_plan, xs_pred.chunk(len(current_plan), 0)):
                xs[:, frames] = pred[:, : len(frames)]
                context_mask[:, frames] = True
        pbar.close()
        return xs

    def _predict_sequence(
        self,
        context: torch.Tensor,
        length: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        history_guidance: Optional[HistoryGuidance] = None,
        sliding_context_len: Optional[int] = None,
        return_all: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict a sequence given context tokens at the beginning, using sliding window if necessary.
        Args
        ----
        context: torch.Tensor, Shape (batch_size, init_context_len, *self.x_shape)
            Initial context tokens to condition on
        length: Optional[int]
            Desired number of tokens in sampled sequence.
            If None, fall back to to self.max_tokens, and
            If bigger than self.max_tokens, sliding window sampling will be used.
        conditions: Optional[torch.Tensor], Shape (batch_size, conditions_len, ...)
            Unprocessed external conditions for sampling, e.g. action or text, optional
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        reconstruction_guidance: float
            Scale of reconstruction guidance (from Video Diffusion Models Ho. et al.)
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        sliding_context_len: Optional[int]
            Max context length when using sliding window. -1 to use max_tokens - 1.
            Has no influence when length <= self.max_tokens as no sliding window is needed.
        return_all: bool
            Whether to return all steps of the sampling process.

        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Predicted sequence with both context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            Record of all steps of the sampling process
        """
        if length is None:
            length = self.max_tokens*2 # Double for difference
        if sliding_context_len is None:
            if self.max_tokens < length:
                raise ValueError(
                    "when length > max_tokens, sliding_context_len must be specified."
                )
            else:
                sliding_context_len = self.max_tokens - 1
        if sliding_context_len == -1:
            sliding_context_len = self.max_tokens - 1

        batch_size, gt_len, *_ = context.shape
        if sliding_context_len < gt_len:
            raise ValueError(
                "sliding_context_len is expected to be >= length of initial context,"
                f"got {sliding_context_len}. If you are trying to use max context, "
                "consider specifying sliding_context_len=-1."
            )

        chunk_size = self.chunk_size if self.use_causal_mask else self.max_tokens*2
        curr_token = gt_len
        xs_pred = context
        x_shape = self.x_shape
        record = None

        n_run = 1 + (length - sliding_context_len - 1) // (self.max_tokens*2 - sliding_context_len)
        pbar = tqdm(
            total=self.sampling_timesteps * n_run,
            initial=0,
            desc="Predicting with DFoT",
            leave=False,
        )
        while curr_token < length:
            if record is not None:
                raise ValueError("return_all is not supported if using sliding window.")
            # actual context depends on whether it's during sliding window or not
            # corner case at the beginning
            c = min(sliding_context_len, curr_token)
            # try biggest prediction chunk size
            h = min(length - curr_token, self.max_tokens*2 - c)
            # chunk_size caps how many future tokens are diffused at once to save compute for causal model
            h = min(h, chunk_size) if chunk_size > 0 else h
            l = c + h
            pad = torch.zeros((batch_size, h, *x_shape))
            # context is last c tokens out of the sequence of generated/gt tokens
            # pad to length that's required by _sample_sequence
            context = torch.cat([xs_pred[:, -c:], pad.to(self.device)], 1)
            # calculate number of model generated tokens (not GT context tokens)
            generated_len = curr_token - max(curr_token - c, gt_len)
            # make context mask
            context_mask = torch.ones((batch_size, c), dtype=torch.long)
            if generated_len > 0:
                context_mask[:, -generated_len:] = 2
            pad = torch.zeros((batch_size, h), dtype=torch.long)
            context_mask = torch.cat([context_mask, pad.long()], 1).to(context.device)

            cond_len = l if self.use_causal_mask else self.max_tokens * 2
            cond_slice = None
            if conditions is not None:
                match self.external_cond_type:
                    case "label":
                        cond_slice = conditions
                    case "action":
                        cond_slice = conditions[:, curr_token - c : curr_token - c + cond_len]
                    case _:
                        raise ValueError(
                            f"Unknown external condition type: {self.external_cond_type}. "
                            "Supported types are 'label' and 'action'."
                        )

            if self.cfg.refinement_sampling.enabled:
                new_pred, record = self._sample_sequence_refine(
                    batch_size,
                    length=l,
                    context=context,
                    context_mask=context_mask,
                    conditions=cond_slice,
                    goback_length=self.cfg.refinement_sampling.goback_length,
                    n_goback=self.cfg.refinement_sampling.n_goback,
                    guidance_fn=guidance_fn,
                    reconstruction_guidance=reconstruction_guidance,
                    history_guidance=history_guidance,
                    return_all=return_all,
                    pbar=pbar,
                )
            else:
                new_pred, record = self._sample_sequence(
                    batch_size,
                    length=l,
                    context=context,
                    context_mask=context_mask,
                    conditions=cond_slice,
                    guidance_fn=guidance_fn,
                    reconstruction_guidance=reconstruction_guidance,
                    history_guidance=history_guidance,
                    return_all=return_all,
                    pbar=pbar,
                )
            xs_pred = torch.cat([xs_pred, new_pred[:, -h:]], 1)
            curr_token = xs_pred.shape[1]
        pbar.close()
        return xs_pred, record

    def _sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        history_guidance: Optional[HistoryGuidance] = None,
        return_all: bool = False,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The unified sampling method, with length up to maximum token size.
        context of length can be provided along with a mask to achieve conditioning.

        Args
        ----
        batch_size: int
            Batch size of the sampling process
        length: Optional[int]
            Number of frames in sampled sequence
            If None, fall back to length of context, and then fall back to `self.max_tokens`
        context: Optional[torch.Tensor], Shape (batch_size, length, *self.x_shape)
            Context tokens to condition on. Assumed to be same across batch.
            Tokens that are specified as context by `context_mask` will be used for conditioning,
            and the rest will be discarded.
        context_mask: Optional[torch.Tensor], Shape (batch_size, length)
            Mask for context
            0 = To be generated, 1 = Ground truth context, 2 = Generated context
            Some sampling logic may discriminate between ground truth and generated context.
        conditions: Optional[torch.Tensor], Shape (batch_size, length (causal) or self.max_tokens (noncausal), ...)
            Unprocessed external conditions for sampling
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        return_all: bool
            Whether to return all steps of the sampling process
        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Complete sequence containing context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            All recorded intermediate results during the sampling process
        """
        x_shape = self.x_shape

        if length is None:
            length = self.max_tokens*2 if context is None else context.shape[1]
        if length > self.max_tokens*2:
            raise ValueError(
                f"length is expected to <={self.max_tokens}, got {length}."
            )
        if context is not None:
            if context_mask is None:
                raise ValueError("context_mask must be provided if context is given.")
            if context.shape[0] != batch_size:
                raise ValueError(
                    f"context batch size is expected to be {batch_size} but got {context.shape[0]}."
                )
            if context.shape[1] != length:
                raise ValueError(
                    f"context length is expected to be {length} but got {context.shape[1]}."
                )
            if tuple(context.shape[2:]) != tuple(x_shape):
                raise ValueError(
                    f"context shape not compatible with x_stacked_shape {x_shape}."
                )

        if context_mask is not None:
            if context is None:
                raise ValueError("context must be provided if context_mask is given. ")
            if context.shape[:2] != context_mask.shape:
                raise ValueError("context and context_mask must have the same shape.")

        if conditions is not None and self.external_cond_type == "action":
            if self.use_causal_mask and conditions.shape[1] != length:
                raise ValueError(
                    f"for causal models, conditions length is expected to be {length}, got {conditions.shape[1]}."
                )
            elif not self.use_causal_mask and conditions.shape[1] != self.max_tokens*2:
                raise ValueError(
                    f"for noncausal models, conditions length is expected to be {self.max_tokens*2}, got {conditions.shape[1]}."
                )

        horizon = length if self.use_causal_mask else self.max_tokens * 2
        padding = horizon - length
        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, horizon, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        # if context is None, create empty context and context mask
        if context is None:
            # create empty context and zero context mask
            context = torch.zeros_like(xs_pred)
            context_mask = torch.zeros_like(
                (batch_size, horizon), dtype=torch.long, device=self.device
            )
        # if context is provided, check its length and pad if necessary
        elif padding > 0:
            # pad context and context mask to reach horizon
            context_pad = torch.zeros(
                (batch_size, padding, *x_shape), device=self.device
            )
            # NOTE: In context mask, -1 = padding, 0 = to be generated, 1 = GT context, 2 = generated context
            context_mask_pad = -torch.ones(
                (batch_size, padding), dtype=torch.long, device=self.device
            )
            context = torch.cat([context, context_pad], 1)
            context_mask = torch.cat([context_mask, context_mask_pad], 1)

        if history_guidance is None:
            # by default, use conditional sampling
            history_guidance = HistoryGuidance.conditional(
                timesteps=self.timesteps,
            )

        # replace xs_pred's context frames with context
        xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

        # generate scheduling matrix
        scheduling_matrix = self._generate_scheduling_matrix(
            horizon - padding,
            padding,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)
        # Full sequence training: fill context tokens' noise levels as -1 in scheduling matrix
        if not self.is_full_sequence:
            scheduling_matrix = torch.where(
                context_mask[None] >= 1, -1, scheduling_matrix
            )

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        record = [] if return_all else None

        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with DFoT",
                leave=False,
            )            

        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            # update context mask by changing 0 -> 2 for fully generated tokens
            context_mask = torch.where(
                torch.logical_and(context_mask == 0, from_noise_levels == -1),
                2,
                context_mask,
            )

            # create a backup with all context tokens unmodified
            xs_pred_prev = xs_pred.clone()
            if return_all:
                record.append(xs_pred.clone())

            conditions_mask = None
            with history_guidance(context_mask) as history_guidance_manager:
                nfe = history_guidance_manager.nfe
                pbar.set_postfix(NFE=nfe)
                xs_pred, from_noise_levels, to_noise_levels, conditions_mask = (
                    history_guidance_manager.prepare(
                        xs_pred,
                        from_noise_levels,
                        to_noise_levels,
                        replacement_fn=self.diffusion_model.q_sample,
                        replacement_only=self.is_full_sequence,
                    )
                )

                if reconstruction_guidance > 0:

                    def composed_guidance_fn(
                        xk: torch.Tensor,
                        pred_x0: torch.Tensor,
                        alpha_cumprod: torch.Tensor,
                    ) -> torch.Tensor:
                        loss = (
                            F.mse_loss(pred_x0, context, reduction="none")
                            * alpha_cumprod.sqrt()
                        )
                        _context_mask = rearrange(
                            context_mask.bool(),
                            "b t -> b t" + " 1" * len(x_shape),
                        )
                        # scale inversely proportional to the number of context frames
                        loss = torch.sum(
                            loss
                            * _context_mask
                            / _context_mask.sum(dim=1, keepdim=True).clamp(min=1),
                        )
                        likelihood = -reconstruction_guidance * 0.5 * loss
                        return likelihood

                else:
                    composed_guidance_fn = guidance_fn

                # update xs_pred by DDIM or DDPM sampling
                xs_pred = self.diffusion_model.sample_step(
                    xs_pred,
                    from_noise_levels,
                    to_noise_levels,
                    repeat(conditions, "b ... -> (b nfe) ...", nfe=nfe).clone() if conditions is not None else None,
                    conditions_mask,
                    guidance_fn=composed_guidance_fn,
                )
                xs_pred = history_guidance_manager.compose(xs_pred)

            # only replace the tokens being generated (revert context tokens)
            xs_pred = torch.where(
                self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev
            )
            pbar.update(1)

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, record
    
    def _sample_sequence_refine(
        self,
        batch_size: int,
        goback_length: int,
        n_goback: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        history_guidance: Optional[HistoryGuidance] = None,
        return_all: bool = False,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError(
            "Refinement sampling is not implemented for DFoT. "
            "Please use _sample_sequence instead."
        )

    