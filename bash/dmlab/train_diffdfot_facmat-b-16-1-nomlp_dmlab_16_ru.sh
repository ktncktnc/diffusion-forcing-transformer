python -m main \
    '+name=Diffv2DFOT_FacMat-B-16-1-NoMLP_DMLAB_16_RU' \
    'experiment=video_generation' \
    'algorithm=difference_dfot_video' \
    \
    dataset=dmlab \
    dataset.max_frames=16 \
    \
    dataset.latent.suffix=17035ae5 \
    dataset.latent.shape=null \
    \
    algorithm/backbone=difference_dit3d_factorized_matrix \
    @FacMatDiT/group_B/B-16-1 \
    algorithm.backbone.spatial_mlp_ratio=0.0 \
    algorithm.backbone.patch_size=2 \
    \
    algorithm/vae=dc_ae_preprocessor \
    algorithm.vae.pretrained_path=/scratch/s224075134/temporal_diffusion/FAR/pretrained/dcae/DCAE_DMLab_Res64-17035ae5.pth \
    \
    'algorithm.noise_level=random_uniform' \
    'algorithm.variable_context.enabled=True' \
    \
    experiment.training.batch_size=32 \
    experiment.validation.batch_size=32 \
    experiment.training.max_steps=500000 \
    experiment.training.optim.accumulate_grad_batches=1 \
    \
    cluster=a2i2_singlegpu \
    cluster.params.gpu_type=h100 