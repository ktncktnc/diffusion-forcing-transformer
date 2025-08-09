python main.py \
    '+name=DiffDFOT_FacMat-B_DMLAB_16_RU' \
    'algorithm=difference_dfot_video' \
    'experiment=video_generation' \
    \
    dataset=dmlab \
    dataset.max_frames=16 \
    # TODO
    dataset.latent.suffix=17035ae5 \
    dataset.latent.shape=null \
    \
    algorithm/backbone=difference_dit3d_factorized_matrix \
    '@FacMatDiT/B' \
    algorithm.backbone.patch_size=2 \
    \
    algorithm/vae=dc_ae_preprocessor \
    algorithm.vae.pretrained_path=/scratch/s224075134/temporal_diffusion/FAR/pretrained/dcae/DCAE_DMLab_Res64-17035ae5.pth \
    algorithm.vae.batch_size=2 \
    \
    algorithm.noise_level=random_uniform \
    algorithm.variable_context.enabled=True \
    \
    experiment.training.batch_size=32 \
    experiment.validation.batch_size=23 \
    experiment.training.max_steps=500000 \
    experiment.training.optim.accumulate_grad_batches=1 \
    \
    wandb.mode=disabled