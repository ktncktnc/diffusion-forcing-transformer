from huggingface_hub import snapshot_download, hf_hub_download

# dataset_url = {
#     "ucf101": "guyuchao/UCF101",
#     # "bair": "guyuchao/BAIR",
#     # "minecraft": "guyuchao/Minecraft",
#     # "minecraft_latent": "guyuchao/Minecraft_Latent",
#     # "dmlab": "guyuchao/DMLab",
#     # "dmlab_latent": "guyuchao/DMLab_Latent"
# }

# for key, url in dataset_url.items():
#     snapshot_download(
#         repo_id=url,
#         repo_type="dataset",
#         local_dir=f"/scratch/s224075134/temporal_diffusion/datasets/{key}",
#         token="xxx"
#     )


hf_hub_download(
    repo_id="guyuchao/FAR_Models",
    filename="dcae/DCAE_Minecraft_Res128-a5677f66.pth",
    local_dir="/scratch/s224075134/temporal_diffusion/FAR/pretrained/dcae"
)