import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from experiments import build_experiment

with initialize(version_base=None, config_path="configurations"):
    cfg = compose(
        config_name="config", 
        overrides=[
            'algorithm=contrastive_dfot_video', 
            "experiment=video_generation",
            'dataset=ucf_101',
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

    # cfg = OmegaConf.to_yaml(cfg)

# print(cfg)

experiment = build_experiment(cfg, None, None)
for task in cfg.experiment.tasks:
    experiment.exec_task(task)

python -m main +name=UCF_101 dataset=ucf_101 algorithm=contrastive_dfot_video experiment=video_generation wandb.mode=disabled