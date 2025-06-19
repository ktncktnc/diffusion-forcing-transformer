from typing import Optional, Any, Dict, Literal, Callable, Tuple
from functools import partial
from omegaconf import DictConfig
import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.utilities import grad_norm
from einops import rearrange, repeat, reduce
from transformers import get_scheduler
from tqdm import tqdm
from algorithms.common.base_pytorch_video_algo import BaseVideoAlgo
from algorithms.common.metrics.video import VideoMetric, SharedVideoMetricModelRegistry
from algorithms.vae import ImageVAE, VideoVAE, MyAutoencoderDC, AutoencoderKL
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print, is_rank_zero
from utils.torch_utils import bernoulli_tensor
from .diffusion import (
    DiscreteDiffusion,
    ContinuousDiffusion,
)
from utils.logging_utils import log_video
from .history_guidance import HistoryGuidance


class ReferenceDFoTVideo(BaseVideoAlgo):
    """
    An algorithm for training and evaluating
    Diffusion Forcing Transformer (DFoT) for video generation.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _build_model(self) -> None:
        diffusion_cls = (
            ContinuousDiffusion
            if self.cfg.diffusion.is_continuous
            else DiscreteDiffusion
        )
        super()._build_model(diffusion_cls=diffusion_cls)

    @property
    def max_frames(self) -> int:
        """Minus 1 since the first frame is always the context frame."""
        return self.cfg.max_frames - 1

    # ---------------------------------------------------------------------
    # NOTE: n_{frames, tokens} indicates the number of frames/tokens
    # that the model actually processes during training/validation.
    # During validation, it may be different from max_{frames, tokens},
    # ---------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return self.max_frames if self.trainer.training else self.cfg.n_frames - 1

    # ---------------------------------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------------------------------
    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        xs = batch["xs"]
        conditions = batch.get("conditions")
        reference = batch.get("reference")
        gt_videos = batch.get("gt_videos")

        if self.external_cond_type == "label":
            conditions = conditions
        elif self.external_cond_type == "action":
            conditions = conditions[:, 1:, ...]

        reference = xs[:, 0, ...]
        xs = xs[:, 1:, ...]
        gt_videos = gt_videos[:, 1:, ...] if gt_videos is not None else None
        masks = torch.ones(*xs.shape[:2]).bool().to(self.device)

        return {
            "xs": xs,
            "conditions": conditions,
            "reference": reference,
            "masks": masks,
            "gt_videos": gt_videos,
        }

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        """
        Training step
        Use first frame or context_frames as reference.
        """
        xs = batch["xs"]
        conditions = batch.get("conditions")
        reference = batch["reference"]
        masks = batch["masks"]

        if self.cfg.reference.predict_difference:
            xs = xs - repeat(reference.unsqueeze(1), 'b n ... -> b (n n_frame) ...', n_frame=xs.shape[1])

        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        xs_pred, loss = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
            reference=reference,
        )
        loss = self._reweight_loss(loss, masks)

        if batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(
                f"{namespace}/loss",
                loss,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
                add_dataloader_idx=False
            )

        if self.cfg.reference.predict_difference:
            xs = xs + repeat(reference.unsqueeze(1), 'b n ... -> b (n n_frame) ...', n_frame=xs.shape[1])
            xs_pred = xs_pred + repeat(reference.unsqueeze(1), 'b n ... -> b (n n_frame) ...', n_frame=xs.shape[1])

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
    def _sample_all_videos(
        self, batch, batch_idx, namespace="validation", n_context_tokens=None
    ) -> Optional[Dict[str, Tensor]]:
        # xs, conditions, reference, *_, gt_videos = batch
        xs = batch["xs"]
        conditions = batch.get("conditions")
        reference = batch["reference"]
        gt_videos = batch["gt_videos"]
        n_context_tokens = n_context_tokens if n_context_tokens is not None else self.n_context_tokens
        all_videos: Dict[str, Tensor] = {"gt": xs}
        
        if self.cfg.reference.predict_difference:
            xs = xs - repeat(reference.unsqueeze(1), 'b n ... -> b (n n_frame) ...', n_frame=xs.shape[1])

        for task in self.tasks:
            sample_fn = (
                self._predict_videos
                if task == "prediction"
                else self._interpolate_videos
            )
            generated = sample_fn(xs, reference=reference, conditions=conditions, n_context_tokens=n_context_tokens)
            if self.cfg.reference.predict_difference:
                generated = generated + repeat(reference.unsqueeze(1), 'b n ... -> b (n n_frame) ...', n_frame=xs.shape[1])
            all_videos[task] = generated

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
        return all_videos

    def _predict_videos(
        self, xs: Tensor, reference: Tensor, n_context_tokens: int, conditions: Optional[Tensor] = None
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
        ).unique()  # context frames are always keyframes

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
            reference=reference,
            conditions=key_conditions,
            history_guidance=history_guidance,
            reconstruction_guidance=self.cfg.diffusion.reconstruction_guidance,
            sliding_context_len=self.cfg.tasks.prediction.sliding_context_len
            or self.max_tokens // 2,
        )
        xs_pred[:, keyframe_indices] = xs_pred_key

        # 2. (Optional) Interpolate the intermediate frames
        if len(keyframe_indices) < xs_pred.shape[1]:
            context_mask = torch.zeros(xs_pred.shape[:2], device=self.device).bool()
            context_mask[:, keyframe_indices] = True
            xs_pred = self._interpolate_videos(
                context=xs_pred,
                reference=reference,
                context_mask=context_mask,
                conditions=conditions,
            )

        return xs_pred


    def _interpolate_videos(
        self,
        context: Tensor,
        reference: Tensor,
        context_mask: Optional[Tensor] = None,
        conditions: Optional[Tensor] = None,
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
                        reference=reference,
                        goback_length=self.cfg.refinement_sampling.goback_length,
                        n_goback=self.cfg.refinement_sampling.n_goback,
                        context_mask=current_context_mask_chunk.long(),
                        conditions=current_conditions_chunk,
                        # history_guidance=history_guidance,
                        pbar=pbar,
                    )
                else:
                    xs_pred_chunk, _ = self._sample_sequence(
                        batch_size=batch_size,
                        reference=reference,
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

    # ---------------------------------------------------------------------
    # Training Utils
    # ---------------------------------------------------------------------

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

    # ---------------------------------------------------------------------
    # Sampling Utils
    # ---------------------------------------------------------------------

    def _predict_sequence(
        self,
        context: torch.Tensor,
        reference: torch.Tensor,
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
            length = self.max_tokens
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

        chunk_size = self.chunk_size if self.use_causal_mask else self.max_tokens

        curr_token = gt_len
        xs_pred = context
        x_shape = self.x_shape
        record = None
        pbar = tqdm(
            total=self.sampling_timesteps
            * (
                1
                + (length - sliding_context_len - 1)
                // (self.max_tokens - sliding_context_len)
            ),
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
            h = min(length - curr_token, self.max_tokens - c)
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

            cond_len = l if self.use_causal_mask else self.max_tokens
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
                    reference=reference,
                    context=context,
                    context_mask=context_mask,
                    conditions=cond_slice,
                    goback_length=self.cfg.refinement_sampling.goback_length,
                    n_goback=self.cfg.refinement_sampling.n_goback,
                    guidance_fn=guidance_fn,
                    reconstruction_guidance=reconstruction_guidance,
                    return_all=return_all,
                    pbar=pbar,
                )
            else:
                new_pred, record = self._sample_sequence(
                    batch_size,
                    length=l,
                    reference=reference,
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
        reference: torch.Tensor,
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
            length = self.max_tokens if context is None else context.shape[1]
        if length > self.max_tokens:
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

        # if conditions is not None:
        #     if self.use_causal_mask and conditions.shape[1] != length:
        #         raise ValueError(
        #             f"for causal models, conditions length is expected to be {length}, got {conditions.shape[1]}."
        #         )
        #     elif not self.use_causal_mask and conditions.shape[1] != self.max_tokens:
        #         raise ValueError(
        #             f"for noncausal models, conditions length is expected to be {self.max_tokens}, got {conditions.shape[1]}."
        #         )

        horizon = length if self.use_causal_mask else self.max_tokens
        padding = horizon - length
        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, horizon, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        if context is None:
            # create empty context and zero context mask
            context = torch.zeros_like(xs_pred)
            context_mask = torch.zeros_like(
                (batch_size, horizon), dtype=torch.long, device=self.device
            )
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
                    reference,
                    from_noise_levels,
                    to_noise_levels,
                    self._process_conditions(
                        (
                            repeat(
                                conditions,
                                "b ... -> (b nfe) ...",
                                nfe=nfe,
                            ).clone()
                            if conditions is not None
                            else None
                        ),
                        from_noise_levels,
                    ),
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
        reference: torch.Tensor,
        goback_length: int,
        n_goback: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        # history_guidance: Optional[HistoryGuidance] = None,
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
            length = self.max_tokens if context is None else context.shape[1]
        if length > self.max_tokens:
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

        horizon = length if self.use_causal_mask else self.max_tokens
        padding = horizon - length
        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, horizon, *x_shape),
            device=self.device,
            generator=self.generator,
        )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        if context is None:
            # create empty context and zero context mask
            context = torch.zeros_like(xs_pred)
            context_mask = torch.zeros_like(
                (batch_size, horizon), dtype=torch.long, device=self.device
            )
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

        # replace xs_pred's context frames with context
        # xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

        # generate scheduling matrix
        # scheduling_matrix = self._generate_refine_scheduling_matrix(
        #     horizon=horizon - padding,
        #     goback_length=goback_length,
        #     n_goback=n_goback,
        #     padding=padding,
        # )        
        scheduling_matrix = self._generate_scheduling_matrix(
            horizon=horizon - padding,
            padding=padding,
        )
        # xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)
        # torch.set_printoptions(threshold=100000000, linewidth=10000)
        # print('\n')
        # print(scheduling_matrix)
        # exit(0)
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)

        # if not self.is_full_sequence:
        #     scheduling_matrix = torch.where(
        #         context_mask[None] >= 1, -1, scheduling_matrix
        #     )
        
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
            to_noise_levels = scheduling_matrix[m+1]
            
            # context_mask = torch.where(
            #     torch.logical_and(context_mask == 0, from_noise_levels == -1),
            #     2,
            #     context_mask,
            # )
            
            # Reverse process: x_tm1 ~ p(x_tm1|x_t)
            # TODO: replace x_tm1 = context_mask * x^c_tm1 + (1-context_mask) * x^g_tm1 
            # print('\n', from_noise_levels[0,0].item(), to_noise_levels[0,0].item())
            # if from_noise_levels[0,0].item() > to_noise_levels[0,0].item():
            if True:
                # print(from_noise_levels, to_noise_levels)
                # create a backup with all context tokens unmodified
                xs_pred_prev = xs_pred.clone()
                if return_all:
                    record.append(xs_pred.clone())

                conditions_mask = None
                # update xs_pred by DDIM or DDPM sampling
                xs_pred = self.diffusion_model.sample_step(
                    xs_pred,
                    reference,
                    from_noise_levels,
                    to_noise_levels,
                    self._process_conditions(
                        (
                            repeat(
                                conditions,
                                "b ... -> (b nfe) ...",
                                nfe=1,
                            ).clone()
                            if conditions is not None
                            else None
                        ),
                        from_noise_levels,
                    ),
                    conditions_mask,
                    guidance_fn=guidance_fn,
                )
                # xc_t = self.diffusion_model.q_sample(context, to_noise_levels)
                # xs_pred = torch.where(
                #     self._extend_x_dim(context_mask) == 0, xs_pred, xc_t
                # )

                pbar.update(1)
            
            # Forward process: x_t ~ p(x_t|x_tm1)
            # else:
            #     xs_pred = self.diffusion_model.q_sample_from_x_k(
            #         xs_pred,
            #         from_noise_levels,
            #         to_noise_levels
            #     )

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, record