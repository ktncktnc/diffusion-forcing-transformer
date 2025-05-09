"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Tuple
import contextlib
import copy
import os
import threading
from typing import Any, Dict, Iterable
import torch
from lightning import Trainer, LightningModule
from lightning.pytorch import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.trainer.states import TrainerFn
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print


class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).

    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.

    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        validate_original_weights: Validate the original weights, as apposed to the EMA weights.
        every_n_steps: Apply EMA every N steps.
        cpu_offload: Offload weights to CPU.

    Adapted from:
    - https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py
    - https://github.com/BioinfoMachineLearning/bio-diffusion/blob/e4bad15139815e562a27fb94dab0c31907522bc5/src/utils/__init__.py
    """

    def __init__(
        self,
        enable: bool = True,
        decay: float = 0.0,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
        optimizer_indices: Tuple[int] | None = None,
    ):
        self.enable = enable
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload
        self.optimizer_indices = optimizer_indices
        if not self.enable:
            return

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enable:
            return
        device = pl_module.device if not self.cpu_offload else torch.device("cpu")
        trainer.optimizers = [
            (
                EMAOptimizer(
                    optim,
                    device=device,
                    decay=self.decay,
                    every_n_steps=self.every_n_steps,
                    current_step=trainer.global_step,
                )
                if not isinstance(optim, EMAOptimizer)
                and (self.optimizer_indices is None or i in self.optimizer_indices)
                else optim
            )
            for i, optim in enumerate(trainer.optimizers)
        ]

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if (
            self.enable
            and stage != TrainerFn.FITTING
            and not self.validate_original_weights
        ):
            # if not fitting, there's no optimizer so we need to manually put ema weights to checkpoint['state_dict']
            # this should be handled at lightning module level
            if not hasattr(pl_module, "should_validate_ema_weights"):
                rank_zero_print(
                    cyan("WARNING: this pl_module is incompatible with EMA callback"),
                    "You will be validating with the original weights, not the EMA weights.",
                )
            pl_module.should_validate_ema_weights = True

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def _should_validate_ema_weights(self, trainer: Trainer) -> bool:
        return not self.validate_original_weights and self._ema_initialized(trainer)

    def _ema_initialized(self, trainer: Trainer) -> bool:
        return any(
            isinstance(optimizer, EMAOptimizer) for optimizer in trainer.optimizers
        )

    def swap_model_weights(self, trainer: Trainer, saving_ema_model: bool = False):
        for optimizer in trainer.optimizers:
            if isinstance(optimizer, EMAOptimizer):
                optimizer.switch_main_parameter_weights(saving_ema_model)

    @contextlib.contextmanager
    def save_ema_model(self, trainer: Trainer):
        """
        Saves an EMA copy of the model + EMA optimizer states for resume.
        """
        self.swap_model_weights(trainer, saving_ema_model=True)
        try:
            yield
        finally:
            self.swap_model_weights(trainer, saving_ema_model=False)

    @contextlib.contextmanager
    def save_original_optimizer_state(self, trainer: Trainer):
        for optimizer in trainer.optimizers:
            if isinstance(optimizer, EMAOptimizer):
                optimizer.save_original_optimizer_state = True
        try:
            yield
        finally:
            for optimizer in trainer.optimizers:
                optimizer.save_original_optimizer_state = False

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """
        Enable loading EMA-enabled optimizer states to EMA-disabled optimizer states.
        """
        if not self.enable:
            optimizer_states = checkpoint["optimizer_states"]
            new_optimizer_states = []
            for optimizer_state in optimizer_states:
                new_optimizer_states.append(
                    optimizer_state["opt"]
                    if "opt" in optimizer_state
                    else optimizer_state
                )
            checkpoint["optimizer_states"] = new_optimizer_states


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple,
        current_model_tuple,
        alpha=(1.0 - decay),
    )


def run_ema_update_cpu(
    ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None
):
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()

    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""
    EMAOptimizer is a wrapper for torch.optim.Optimizer that computes
    Exponential Moving Average of parameters registered in the optimizer.

    EMA parameters are automatically updated after every step of the optimizer
    with the following formula:

        ema_weight = decay * ema_weight + (1 - decay) * training_weight

    To access EMA parameters, use ``swap_ema_weights()`` context manager to
    perform a temporary in-place swap of regular parameters with EMA
    parameters.

    Notes:
        - EMAOptimizer is not compatible with APEX AMP O2.

    Args:
        optimizer (torch.optim.Optimizer): optimizer to wrap
        device (torch.device): device for EMA parameters
        decay (float): decay factor

    Returns:
        returns an instance of torch.optim.Optimizer that computes EMA of
        parameters

    Example:
        model = Model().to(device)
        opt = torch.optim.Adam(model.parameters())

        opt = EMAOptimizer(opt, device, 0.9999)

        for epoch in range(epochs):
            training_loop(model, opt)

            regular_eval_accuracy = evaluate(model)

            with opt.swap_ema_weights():
                ema_eval_accuracy = evaluate(model)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.9999,
        every_n_steps: int = 1,
        current_step: int = 0,
    ):
        self.optimizer = optimizer
        self.decay = decay
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps
        self.save_original_optimizer_state = False

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None

        self.ema_params = ()
        self.in_saving_ema_model_context = False

    def all_parameters(self) -> Iterable[torch.Tensor]:
        return (param for group in self.param_groups for param in group["params"])

    def step(self, closure=None, grad_scaler=None, **kwargs):
        self.join()

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()

            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())

            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device)
                for param in opt_params[len(self.ema_params) :]
            )
            self.rebuild_ema_params = False

        if (
            getattr(self.optimizer, "_step_supports_amp_scaling", False)
            and grad_scaler is not None
        ):
            loss = self.optimizer.step(closure=closure, grad_scaler=grad_scaler)
        else:
            loss = self.optimizer.step(closure)

        if self._should_update_at_step():
            self.update()
        self.current_step += 1
        return loss

    def _should_update_at_step(self) -> bool:
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self):
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True)
                for param in self.all_parameters()
            )

            if self.device.type == "cuda":
                ema_update(self.ema_params, current_model_state, self.decay)

        if self.device.type == "cpu":
            self.thread = threading.Thread(
                target=run_ema_update_cpu,
                args=(
                    self.ema_params,
                    current_model_state,
                    self.decay,
                    self.stream,
                ),
            )
            self.thread.start()

    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False):
        self.join()
        self.in_saving_ema_model_context = saving_ema_model
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        r"""
        A context manager to in-place swap regular parameters with EMA
        parameters.
        It swaps back to the original regular parameters on context manager
        exit.

        Args:
            enabled (bool): whether the swap should be performed
        """

        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def join(self):
        if self.stream is not None:
            self.stream.synchronize()

        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        self.join()

        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()

        # if we are in the context of saving an EMA model, the EMA weights are in the modules' actual weights
        ema_params = (
            self.ema_params
            if not self.in_saving_ema_model_context
            else list(self.all_parameters())
        )
        state_dict = {
            "opt": self.optimizer.state_dict(),
            "ema": ema_params,
            "current_step": self.current_step,
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.join()

        if "opt" in state_dict:
            self.optimizer.load_state_dict(state_dict["opt"])
            self.ema_params = tuple(
                param.to(self.device) for param in copy.deepcopy(state_dict["ema"])
            )
            self.current_step = state_dict["current_step"]
            self.decay = state_dict["decay"]
            self.every_n_steps = state_dict["every_n_steps"]
        else:  # loading non-EMA state dict
            self.optimizer.load_state_dict(state_dict)
            self.ema_params = tuple(
                copy.deepcopy(param.data.detach()).to(self.device)
                for param in self.all_parameters()
            )
        self.rebuild_ema_params = False

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True
