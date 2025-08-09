python main.py \
    '+name=DFOT_FacDiT-XL_TAICHIKL_16_RU' \
    'algorithm=dfot_video' \
    'experiment=video_generation' \
    \
    dataset=taichi \
    dataset.max_frames=16 \
    dataset.latent.suffix=kl_f8_autoencoder \
    dataset.latent.shape=null \
    \
    algorithm/backbone=dit3d_factorized_attention \
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
    experiment.training.batch_size=8 \
    experiment.validation.batch_size=8 \
    experiment.training.max_steps=200000 \
    experiment.training.optim.accumulate_grad_batches=2 \
    \
    wandb.mode=disabled