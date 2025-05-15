import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict
import sys
sys.path.append('..')
import hydra
import torch
from experiments.video_generation import VideoGenerationExperiment
from einops import rearrange
import numpy as np
from scipy.stats import spearmanr
import cv2

# Initialize Hydra
# The version_base is optional but recommended to avoid deprecation warnings
with initialize(version_base=None, config_path="configurations"):
    # Compose the config - replace "config_name" with your actual config name
    # You can also override values here if needed
    cfg = compose(
        config_name="config", 
        overrides=[
            'algorithm=dfot_video', 
            "experiment=video_generation",
            'dataset=cond_ucf_101',
            'experiment.tasks=[validation]',
            'experiment.validation.data.shuffle=False',
            'dataset.context_length=4',
            'dataset.n_frames=16',
            'experiment.validation.batch_size=4',
            'algorithm.tasks.prediction.history_guidance.name=conditional',
            '++algorithm={backbone: {depth: 12, hidden_size: 768, num_heads: 12}}', 
            'load=/home/s224075134/diffusion-forcing-transformer/outputs/outputs/video_generation/training/cond_ucf_101/dfot_video/2025-05-03/07-07-26/checkpoints/last.ckpt'
            ],
        return_hydra_config=True
    )
    
    # Now you have the config in the 'cfg' variable
    # You can print it, access values, etc. without running your actual application
    # print(OmegaConf.to_yaml(cfg))
    
    # Access config values
    # print(f"Some value from config: {cfg.some_key}")

cfg_choice = cfg['hydra'].runtime.choices
with open_dict(cfg):
    if cfg_choice["experiment"] is not None:
        cfg.experiment._name = cfg_choice["experiment"]
    if cfg_choice["dataset"] is not None:
        cfg.dataset._name = cfg_choice["dataset"]
    if cfg_choice["algorithm"] is not None:
        cfg.algorithm._name = cfg_choice["algorithm"]
    

print(OmegaConf.to_yaml(cfg))

exp = VideoGenerationExperiment(cfg, None, None)
algo = exp._build_algo()
algo = algo.to('cuda:0')