python main.py \
    '+name=DFOT_FacMat-S-64-1-Bias_TAICHIKL_16_Res128_RU' \
    'algorithm=dfot_video' \
    'experiment=video_generation' \
    \
    dataset=taichi \
    dataset.resolution=128 \
    dataset.max_frames=16 \
    dataset.latent.suffix=kl_f8_autoencoder \
    dataset.latent.shape=null \
    \
    algorithm/backbone=dit3d_factorized_matrix \
    '@FacMatDiT/group_S/S-64-1' \
    algorithm.backbone.use_bias=True \
    algorithm.backbone.patch_size=2 \
    \
    algorithm/vae=kl_autoencoder_preprocessor \
    algorithm.vae.pretrained_path=stabilityai/sd-vae-ft-ema \
    algorithm.vae.batch_size=16 \
    \
    algorithm.noise_level=random_uniform \
    algorithm.variable_context.enabled=True \
    \
    experiment.training.batch_size=16 \
    experiment.training.max_steps=200000 \
    experiment.training.optim.accumulate_grad_batches=4 \
    \
    experiment.validation.val_every_n_step=9999999999 \
    \
    cluster=a2i2_multigpu \
    cluster.params.gpu_type=v100 \
    cluster.params.num_gpus=2