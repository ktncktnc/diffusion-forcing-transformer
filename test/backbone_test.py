import sys
sys.path.append('..')
import hydra
import torch
from experiments.video_generation import VideoGenerationExperiment
from einops import rearrange
import numpy as np
from scipy.stats import spearmanr
import cv2


import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict
# Initialize Hydra
# The version_base is optional but recommended to avoid deprecation warnings
with initialize(version_base=None, config_path="../configurations"):
    # Compose the config - replace "config_name" with your actual config name
    # You can also override values here if needed
    cfg = compose(
        config_name="config", 
        overrides=[
            'algorithm=contrastive_dfot_video', 
            "experiment=video_generation",
            'dataset=ucf_101',
            'experiment.tasks=[validation]',
            'experiment.validation.data.shuffle=False',
            'dataset.context_length=4',
            'dataset.n_frames=12',
            'experiment.validation.batch_size=4',
            'algorithm.tasks.prediction.history_guidance.name=vanilla',
            '++algorithm.backbone.name=u_net3d', 
            '++algorithm.backbone.network_size=48', 
            '++algorithm.backbone.num_res_blocks=2', 
            '++algorithm.backbone.resnet_block_groups=8', 
            '++algorithm.backbone.dim_mults=[1, 2, 4, 8]', 
            '++algorithm.backbone.attn_resolutions=[8, 16, 32, 64]', 
            '++algorithm.backbone.attn_dim_head=32', 
            '++algorithm.backbone.attn_heads=4', 
            '++algorithm.backbone.use_linear_attn=True', 
            '++algorithm.backbone.use_init_temporal_attn=True', 
            '++algorithm.backbone.init_kernel_size=7', 
            '++algorithm.backbone.dropout=0.0',
            'algorithm.diffusion.use_causal_mask=False'
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
    

exp = VideoGenerationExperiment(cfg, None, None)
algo = exp._build_algo()
#algo = algo.to('cuda:0')


x = torch.randn(4, 12, 3, 64, 64)
k = torch.randint(0, 1000, (4, 12))
x, h = algo.diffusion_model.model(x, k, return_representation='a')

print(x.shape)
print(h.shape)