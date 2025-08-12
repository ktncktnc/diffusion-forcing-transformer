python main.py \
    '+name=DiffDFOT_FacMat-XL-64-1-NoMLP_TAICHIKL_16_RU' \
    'algorithm=difference_dfot_video' \
    'experiment=video_generation' \
    \
    dataset=taichi \
    dataset.max_frames=16 \
    dataset.latent.suffix=kl_f8_autoencoder \
    dataset.latent.shape=null \
    \
    algorithm/backbone=difference_dit3d_factorized_matrix \
    '@FacMatDiT/group_XL/XL-64-1' \
    algorithm.backbone.spatial_mlp_ratio=0.0 \
    algorithm.backbone.patch_size=2 \
    \
    algorithm/vae=kl_autoencoder_preprocessor \
    algorithm.vae.pretrained_path=stabilityai/sd-vae-ft-ema \
    algorithm.vae.batch_size=8 \
    \
    algorithm.noise_level=random_uniform \
    algorithm.variable_context.enabled=True \
    \
    experiment.training.batch_size=4 \
    experiment.validation.batch_size=4 \
    experiment.training.max_steps=200000 \
    experiment.training.optim.accumulate_grad_batches=4 \
    \
    cluster=a2i2_singlegpu \
    cluster.params.gpu_type=h100