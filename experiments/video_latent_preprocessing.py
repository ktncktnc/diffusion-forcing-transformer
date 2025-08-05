from datasets.video import (
    MinecraftSimpleVideoDataset,
    UCF101SimpleVideoDataset,
    DMLabSimpleVideoDataset,
    BAIRSimpleVideoDataset,
    TaichiSimpleVideoDataset
)
from algorithms.vae import ImageVAEPreprocessor, DCAEPreprocessor, AutoencoderKL, AutoencoderKLPreprocessor, Titok_KLPreprocessor
from .base_exp import BaseLightningExperiment
from .data_modules import ValDataModule
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True

class VideoLatentPreprocessingExperiment(BaseLightningExperiment):
    """
    Experiment for preprocessing videos to latents using a pretrained ImageVAE model.
    """

    compatible_algorithms = dict(
        image_vae_preprocessor=ImageVAEPreprocessor,
        dc_ae_preprocessor=DCAEPreprocessor,
        dc_ae_16x_preprocessor=DCAEPreprocessor,
        kl_autoencoder_preprocessor=AutoencoderKLPreprocessor,
        titok_kl_preprocessor=Titok_KLPreprocessor
    )

    compatible_datasets = dict(
        minecraft=MinecraftSimpleVideoDataset,
        ucf_101=UCF101SimpleVideoDataset,
        dmlab=DMLabSimpleVideoDataset,
        bair=BAIRSimpleVideoDataset,
        taichi=TaichiSimpleVideoDataset,
        test_taichi=TaichiSimpleVideoDataset,
    )

    data_module_cls = ValDataModule

    def training(self) -> None:
        raise NotImplementedError(
            "Training not implemented for video preprocessing experiments"
        )

    def testing(self) -> None:
        raise NotImplementedError(
            "Testing not implemented for video preprocessing experiments"
        )
