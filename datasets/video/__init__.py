from .minecraft import (
    MinecraftAdvancedVideoDataset,
    MinecraftSimpleVideoDataset,
)
from .kinetics_600 import (
    Kinetics600AdvancedVideoDataset,
    Kinetics600SimpleVideoDataset,
)
from .realestate10k import (
    RealEstate10KAdvancedVideoDataset,
    RealEstate10KSimpleVideoDataset,
)
from .realestate10k_ood import RealEstate10KOODAdvancedVideoDataset
from .realestate10k_mini import RealEstate10KMiniAdvancedVideoDataset

from .ucf_101 import UCF101SimpleVideoDataset, UCF101AdvancedVideoDataset
from .dmlab import DMLabSimpleVideoDataset, DMLabAdvancedVideoDataset
from .bair import BAIRSimpleVideoDataset, BAIRAdvancedVideoDataset
