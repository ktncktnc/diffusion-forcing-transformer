python main.py \
    +name=DiffDFOT_FacMat-XL-64-1-Bias_KINETICS-600_17_RU \
    algorithm=difference_dfot_video \
    experiment=video_generation \
    \
    dataset=kinetics_600 \
    dataset.max_frames=17 \
    dataset.latent.suffix=null \
    dataset.latent.shape=null \
    \
    algorithm/backbone=difference_dit3d_factorized_matrix \
    @FacMatDiT/group_XL/XL-64-1 \
    algorithm.backbone.spatial_mlp_ratio=4.0 \
    algorithm.backbone.use_bias=True \
    algorithm.backbone.patch_size=1 \
    \
    algorithm.vae.pretrained_path=pretrained:VideoVAE_K600.ckpt \
    algorithm.vae.batch_size=2 \
    \
    algorithm.noise_level=random_uniform \
    algorithm.variable_context.enabled=True \
    \
    experiment.training.batch_size=16 \
    experiment.training.max_steps=1000000 \
    experiment.training.optim.accumulate_grad_batches=2 \
    experiment.validation.batch_size=8 \
    experiment.validation.limit_batch=25 \
    \
    cluster=a2i2_multigpu \
    cluster.params.gpu_type=l40s \
    cluster.params.num_gpus=4