from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="maxin-cn/Taichi-HD",
    repo_type="dataset",
    local_dir='/scratch/s224075134/temporal_diffusion/datasets/video/hg_taichi',
    local_dir_use_symlinks=False
)