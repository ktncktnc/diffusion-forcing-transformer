python main.py \
    '+name=DFOT_DiT/XL_TAICHIKL_16_RU' \
    'algorithm=dfot_video' \
    'experiment=video_generation' \
    \
    dataset=taichi \
    dataset.max_frames=16 \
    dataset.latent.suffix=kl_f8_autoencoder \
    dataset.latent.shape=null \
    \
    algorithm/backbone=dit3d \
    '@DiT/XL' \
    algorithm.backbone.patch_size=2 \
    \
    algorithm/vae=kl_autoencoder_preprocessor \
    algorithm.vae.pretrained_path=stabilityai/sd-vae-ft-ema \
    algorithm.vae.batch_size=16 \
    \
    algorithm.noise_level=random_uniform \
    algorithm.variable_context.enabled=True \
    \
    experiment.training.batch_size=4 \
    experiment.validation.batch_size=4 \
    experiment.training.max_steps=400000 \
    experiment.training.optim.accumulate_grad_batches=4 \
    \
    resume=/scratch/s224075134/temporal_diffusion/diffusion-forcing-transformer/outputs/video_generation/training/taichi/dfot_video/2025-07-30/09-57-45/checkpoints/checkpoint_100000 \
    cluster=a2i2_singlegpu \
    cluster.params.gpu_type=h100