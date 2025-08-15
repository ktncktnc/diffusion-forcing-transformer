python main.py \
    '+name=DFOT_FacMat-XL-64-1-Bias_TAICHIKL_16_RU' \
    'algorithm=dfot_video' \
    'experiment=video_generation' \
    \
    dataset=taichi \
    dataset.max_frames=16 \
    dataset.latent.suffix=kl_f8_autoencoder \
    dataset.latent.shape=null \
    \
    algorithm/backbone=dit3d_factorized_matrix \
    '@FacMatDiT/group_XL/XL-64-1' \
    algorithm.backbone.spatial_mlp_ratio=4.0 \
    algorithm.backbone.use_bias=True \
    algorithm.backbone.patch_size=2 \
    \
    algorithm/vae=kl_autoencoder_preprocessor \
    algorithm.vae.pretrained_path=stabilityai/sd-vae-ft-ema \
    algorithm.vae.batch_size=2 \
    \
    algorithm.noise_level=random_uniform \
    algorithm.variable_context.enabled=True \
    \
    experiment.training.batch_size=8 \
    experiment.validation.batch_size=2 \
    experiment.training.max_steps=200000 \
    experiment.training.optim.accumulate_grad_batches=1 \
    resume=/scratch/s224075134/temporal_diffusion/diffusion-forcing-transformer/outputs/video_generation/training/taichi/dfot_video/2025-08-13/02-41-38/checkpoints/checkpoint_10000 \
    \
    cluster=a2i2_multigpu \
    cluster.params.gpu_type=v100 \
    cluster.params.num_gpus=8