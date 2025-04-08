from algorithms.dfot.backbones.dit.dit3d import DiT3D
from omegaconf import OmegaConf
import torch
# name: dit3d 
# variant: full
# pos_emb_type: rope_3d
# patch_size: 2
# hidden_size: 384
# depth: 12
# num_heads: 6
# mlp_ratio: 4.0
# use_gradient_checkpointing: False

cfg = OmegaConf.create({
    "name": "dit3d",
    "variant": "full",
    "pos_emb_type": "rope_3d",
    "patch_size": 2,
    "hidden_size": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "use_gradient_checkpointing": False
})
import torch

model = DiT3D(
    cfg,
    x_shape=torch.Size([3, 64, 64]),
    max_tokens=512,
    external_cond_dim=0,
    use_causal_mask=False
)

print(model)

x = torch.randn(1, 8, 3, 64, 64)
noise_level = torch.randn(1,8)

out = model(x, noise_level)
print(out.shape)