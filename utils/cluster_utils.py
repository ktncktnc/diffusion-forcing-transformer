"""
utils for submitting to clusters, such as slurm
"""

import os
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from pathlib import Path

from utils.print_utils import cyan

# This is set below.
REPO_DIR = None

def write_slurm_script(file_path, slurm_script):
    with file_path.open("w") as f:
        f.write(slurm_script)


def submit_slurm_job(
    cfg: DictConfig,
    python_args: str,
    org_args: str,
    project_root: Path,
):
    log_dir = project_root / "slurm_logs" / f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{cfg.name}"
    log_dir.mkdir(exist_ok=True, parents=True)
    (project_root / "slurm_logs" / "latest").unlink(missing_ok=True)
    (project_root / "slurm_logs" / "latest").symlink_to(log_dir, target_is_directory=True)

    params = dict(name=cfg.name, log_dir=log_dir, project_root=project_root, python_args=python_args)
    params.update(cfg.cluster.params)
    slurm_script = cfg.cluster.launch_template.format(**params)

    org_params = dict(name=cfg.name, log_dir=log_dir, project_root=project_root, python_args=org_args)
    org_params.update(cfg.cluster.params)
    slurm_script_org = cfg.cluster.launch_template.format(**org_params)

    slurm_script_path = log_dir / "job.slurm"
    write_slurm_script(slurm_script_path, slurm_script)
    write_slurm_script(log_dir / "job.org.slurm", slurm_script_org)

    os.system(f"chmod +x {slurm_script_path}")
    os.system(f"sbatch {slurm_script_path}")

    print(f"\n{cyan('script:')} {slurm_script_path}\n{cyan('slurm errors and logs:')} {log_dir}\n")

    return log_dir
