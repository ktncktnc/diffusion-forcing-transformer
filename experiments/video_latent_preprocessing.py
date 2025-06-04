from datasets.video import (
    MinecraftSimpleVideoDataset,
    UCF101SimpleVideoDataset,
    DMLabSimpleVideoDataset,
    BAIRSimpleVideoDataset
)
from algorithms.vae import ImageVAEPreprocessor, DCAEPreprocessor, AutoencoderKL, AutoencoderKLPreprocessor
from .base_exp import BaseLightningExperiment
from .data_modules import ValDataModule


class VideoLatentPreprocessingExperiment(BaseLightningExperiment):
    """
    Experiment for preprocessing videos to latents using a pretrained ImageVAE model.
    """

    compatible_algorithms = dict(
        image_vae_preprocessor=ImageVAEPreprocessor,
        dc_ae_preprocessor=DCAEPreprocessor,
        kl_autoencoder_preprocessor=AutoencoderKLPreprocessor
    )

    compatible_datasets = dict(
        minecraft=MinecraftSimpleVideoDataset,
        ucf_101=UCF101SimpleVideoDataset,
        dmlab=DMLabSimpleVideoDataset,
        bair=BAIRSimpleVideoDataset
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
