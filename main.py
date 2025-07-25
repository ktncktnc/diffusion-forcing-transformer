"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.

Main file for the project. This will create and run new experiments and load checkpoints from wandb. 
Borrowed the wandb code from David Charatan and wandb.ai.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from utils.print_utils import cyan
from utils.ckpt_utils import (
    download_checkpoint,
    download_pretrained,
    is_hf_path,
    is_run_id,
    is_existing_run,
    download_vae_checkpoints,
    parse_load,
    has_checkpoint,
    generate_unexisting_run_id,
    wandb_to_local_path,
)
from utils.cluster_utils import submit_slurm_job
from utils.distributed_utils import rank_zero_print, is_rank_zero
from utils.hydra_utils import unwrap_shortcuts

org_argv = None  # This will be set in the run function.

def run_local(cfg: DictConfig):
    # delay some imports in case they are not needed in non-local envs for submission
    from experiments import build_experiment
    from utils.wandb_utils import OfflineWandbLogger, SpaceEfficientWandbLogger

    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # Get yaml names
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    if is_rank_zero:
        print(cyan(f"Outputs will be saved to:"), output_dir)
        (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (output_dir.parents[1] / "latest-run").symlink_to(
            output_dir, target_is_directory=True
        )

    requeue = cfg.get("requeue", None)
    requeue_path = (
        f"{cfg.wandb.entity}/{cfg.wandb.project}/{requeue}" if requeue else None
    )
    requeue_has_checkpoint = requeue is not None and has_checkpoint(requeue_path)
    requeue_is_existing_run = requeue is not None and is_existing_run(requeue_path)

    # Set up logging with wandb.
    if cfg.wandb.mode != "disabled":
        # If resuming, merge into the existing run on wandb.
        resume = cfg.get("resume", None)
        # if resume and not requeue_is_existing_run:
        #     name = None
        # else:
        if "dfot" in cfg.algorithm._name:
            name = f"{cfg.name}/{cfg.experiment.tasks[0]}: {cfg.experiment._name}/{cfg.dataset._name}/{cfg.algorithm._name}/{cfg.algorithm.backbone.name}/{cfg.algorithm.noise_level} ({output_dir.parent.name}/{output_dir.name})"
        else:
            name = f"{cfg.name}/{cfg.experiment.tasks[0]}: {cfg.experiment._name}/{cfg.dataset._name}/{cfg.algorithm._name} ({output_dir.parent.name}/{output_dir.name})"

        print("name", name)
        if "_on_compute_node" in cfg and cfg.cluster.is_compute_node_offline:
            logger_cls = OfflineWandbLogger
        else:
            logger_cls = SpaceEfficientWandbLogger

        offline = cfg.wandb.mode != "online"
        wandb_kwargs = {
            k: v
            for k, v in OmegaConf.to_container(cfg.wandb, resolve=True).items()
            if k != "mode"
        }
        tags = [cfg.experiment._name, cfg.dataset._name, cfg.algorithm._name] + cfg.experiment.tasks
        if "dfot" in cfg.algorithm._name:
            tags = tags +([f"eval_scheduling_matrix:{cfg.algorithm.scheduling_matrix}", f"train_noise_level:{cfg.algorithm.noise_level}", cfg.algorithm.diffusion.objective])
        if cfg.algorithm._name.startswith('gibbs'):
            tags.append(f"mask_type:{cfg.algorithm.backbone.gibbs.mask_type}")
        # add cfg.algorithm.backbone.name if it exists
        if hasattr(cfg.algorithm, "backbone") and hasattr(cfg.algorithm.backbone, "name"):
            tags.append(cfg.algorithm.backbone.name)
        
        if cfg.dataset.latent.enabled:
            tags.append("latent:enabled")
            tags.append(f"latent_suffix:{cfg.dataset.latent.suffix}")
        else:
            tags.append("latent:disabled")
        tags.append(f"resolution:{cfg.dataset.resolution}")
        
        logger = logger_cls(
            name=name,
            save_dir=str(output_dir),
            offline=offline,
            log_model="all" if not offline else False,
            config=OmegaConf.to_container(cfg),
            # id=resume or requeue,
            tags=tags,
            **wandb_kwargs,
        )
    else:
        logger = None

    # Load ckpt
    resume = cfg.get("resume", None)
    if requeue_has_checkpoint:
        if is_rank_zero:
            print(cyan(f"Resuming from requeued run: {requeue}"))
            download_checkpoint(
                f"{cfg.wandb.entity}/{cfg.wandb.project}/{requeue}",
                Path(os.path.join(cfg.output_dir, 'downloaded')),
                "latest",
            )
        resume = requeue

    load = cfg.get("load", None)
    checkpoint_path = None
    load_id = None
    if resume:
        checkpoint_path = resume
    elif load:
        load_id = parse_load(load)[0]
        if load_id is None:
            checkpoint_path = load

    # if load_id:
    #     run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
    #     checkpoint_path = wandb_to_local_path(run_path)
    # elif load and is_hf_path(load):
    #     checkpoint_path = download_pretrained(load)

    # launch experiment
    experiment = build_experiment(cfg, logger, checkpoint_path)
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)


def run_slurm(cfg: DictConfig, org_argv):
    python_args = (
        " ".join(
            [
                (
                    f"'+requeue={generate_unexisting_run_id(cfg.wandb.entity, cfg.wandb.project)}'"
                    if (arg.startswith("+requeue") and not is_run_id(arg.split("=")[1]))
                    else f"'{arg}'"
                )
                for arg in sys.argv[1:]
            ]
        )
        + " +_on_compute_node=True"
    )
    org_args = (
        " ".join(
            [
                (
                    f"'+requeue={generate_unexisting_run_id(cfg.wandb.entity, cfg.wandb.project)}'"
                    if (arg.startswith("+requeue") and not is_run_id(arg.split("=")[1]))
                    else f"'{arg}'"
                )
                for arg in org_argv[1:]
            ]
        )
    )

    project_root = Path.cwd()
    while not (project_root / ".git").exists():
        project_root = project_root.parent
        if project_root == Path("/"):
            raise Exception("Could not find repo directory!")

    slurm_log_dir = submit_slurm_job(
        cfg,
        python_args,
        org_args,
        project_root,
    )

    if (
        "cluster" in cfg
        and cfg.cluster.is_compute_node_offline
        and cfg.wandb.mode == "online"
    ):
        print(
            "Job submitted to a compute node without internet. This requires manual syncing on login node."
        )
        osh_command_dir = project_root / ".wandb_osh_command_dir"

        osh_proc = None
        # if click.confirm("Do you want us to run the sync loop for you?", default=True):
        osh_proc = subprocess.Popen(["wandb-osh", "--command-dir", osh_command_dir])
        print(f"Running wandb-osh in background... PID: {osh_proc.pid}")
        print(f"To kill the sync process, run 'kill {osh_proc.pid}' in the terminal.")
        print(
            f"You can manually start a sync loop later by running the following:",
            cyan(f"wandb-osh --command-dir {osh_command_dir}"),
        )

    print(
        "Once the job gets allocated and starts running, we will print a command below "
        "for you to trace the errors and outputs: (Ctrl + C to exit without waiting)"
    )
    msg = f"tail -f {slurm_log_dir}/* \n"

    if cfg.cluster.waiting_for_job:
        try:
            while not list(slurm_log_dir.glob("*.out")) and not list(
                slurm_log_dir.glob("*.err")
            ):
                time.sleep(1)
            print(cyan("To trace the outputs and errors, run the following command:"), msg)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting...")
            print(
                cyan(
                    "To trace the outputs and errors, manually wait for the job to start and run the following command:"
                ),
                msg,
            )
    else:
        print(
            cyan(
                "To trace the outputs and errors, manually wait for the job to start and run the following command:"
            ),
            msg,
        )


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig):
    if "_on_compute_node" in cfg and cfg.cluster.is_compute_node_offline:
        with open_dict(cfg):
            if cfg.cluster.is_compute_node_offline and cfg.wandb.mode == "online":
                cfg.wandb.mode = "offline"

    if "name" not in cfg:
        raise ValueError(
            "must specify a name for the run with command line argument '+name=[name]'"
        )

    if not cfg.wandb.get("entity", None):
        raise ValueError(
            "must specify wandb entity in 'configurations/config.yaml' or with command line"
            " argument 'wandb.entity=[entity]' \n An entity is your wandb user name or group"
            " name. This is used for logging. If you don't have an wandb account, please signup at https://wandb.ai/"
        )

    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.name)

    # If resuming or loading a wandb ckpt and not on a compute node, download the checkpoint.
    resume = cfg.get("resume", None)
    load = cfg.get("load", None)
    load_id = None

    if resume and load:
        raise ValueError(
            "When resuming a wandb run with `resume=[wandb id]`, checkpoint will be loaded from the cloud"
            "and `load` should not be specified."
        )

    option = None
    if resume:
        load_id = resume
        option = "latest"
    elif load:
        load_id, option = parse_load(load)
        option = "best" if option is None else option

    if not "skip_download" in cfg:
        # if load_id and "_on_compute_node" not in cfg:
        #     run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        #     download_checkpoint(run_path, Path(os.path.join(cfg.output_dir, 'downloaded')), option=option)
        if "_on_compute_node" not in cfg and is_rank_zero:
            download_vae_checkpoints(cfg)
        if load and is_hf_path(load) and "_on_compute_node" not in cfg:
            download_pretrained(load)

    if "cluster" in cfg and not "_on_compute_node" in cfg:
        print(
            cyan(
                "Slurm detected, submitting to compute node instead of running locally..."
            )
        )
        run_slurm(cfg, org_argv)
    else:
        run_local(cfg)

def set_org_argv(argv):
    """
    Set the global org_argv variable to the provided argv.
    This is used to pass the original command line arguments to the run function.
    """
    global org_argv
    org_argv = argv

if __name__ == "__main__":
    set_org_argv(sys.argv.copy())
    sys.argv = unwrap_shortcuts(sys.argv, config_path="configurations", config_name="config")
    run()  # pylint: disable=no-value-for-parameter
