python main.py \
    '+name=DFOT_TAICHI_16_RI' \
    'algorithm=dfot_video' \
    'experiment=video_generation' \
    \
    dataset=taichi \
    dataset.max_frames=16 \
    dataset.latent.suffix=artitok_taichi500k \
    dataset.latent.shape=[4,1,32] \
    \
    algorithm/backbone=dit3d \
    '@DiT/XL' \
    algorithm.backbone.patch_size=1 \
    \
    algorithm/vae=titok_kl_preprocessor \
    algorithm.vae.pretrained_path=/scratch/s224075134/temporal_diffusion/AR-Diffusion/experiments/taichi_vae/ckpt_dir/checkpoint-500000/model.safetensors \
    algorithm.vae.batch_size=2 \
    \
    algorithm.noise_level=random_independent \
    algorithm.variable_context.enabled=False \
    \
    experiment.training.batch_size=16 \
    experiment.validation.batch_size=16 \
    experiment.training.max_steps=200000 \
    experiment.training.optim.accumulate_grad_batches=2 \
    \
    cluster=a2i2_multigpu \
    cluster.params.gpu_type=h100