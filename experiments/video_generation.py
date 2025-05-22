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
from algorithms.dfot import DFoTVideo, DFoTVideoPose
from algorithms.gibbs_sampling_dfot import GibbsDFoTVideo
from algorithms.reference_dfot import ReferenceDFoTVideo
from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class VideoGenerationExperiment(BaseLightningExperiment):
    """
    A video generation experiment
    """

    compatible_algorithms = dict(
        dfot_video=DFoTVideo,
        dfot_video_pose=DFoTVideoPose,
        gibbs_dfot_video=GibbsDFoTVideo,
        reference_dfot_video=ReferenceDFoTVideo,
        sd_video=DFoTVideo,
        sd_video_3d=DFoTVideoPose,
    )

    compatible_datasets = dict(
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

    data_module_cls = _data_module_cls
