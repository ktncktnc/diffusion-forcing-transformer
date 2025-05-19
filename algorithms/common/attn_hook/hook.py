from algorithms.dfot.backbones.dit.dit_blocks import Attention
import torch 
import os
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np


attn_maps = {}
hooks = []


def hook_function(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module, "attn_map"):
            timestep = module.timestep
            if torch.all(timestep==timestep[0,0].item()):
                timestep = timestep[0,0].item()
            elif torch.all((timestep == 0) | (timestep == 999)):
                timestep = 999 
            else:
                # get any index where it is not equal to 0, 999
                mask = (timestep != 0) & (timestep != 999)
                indices = torch.nonzero(mask, as_tuple=True)
                        
                # Get the first matching element (you can choose any index)
                first_index = (indices[0][0], indices[1][0])
                timestep = timestep[first_index].item()

            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = module.attn_map.cpu() if detach else module.attn_map
            del module.attn_map

    return forward_hook


def register_attention_hook(model: torch.nn.Module, hook_function, target_name):
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue
        
        hook = module.register_forward_hook(hook_function(name))
        module.attn_hook = hook
        module.store_attn_map = True
    
    return model


def clear_hooks(model):
    """
    Clear the hooks from the model.
    """
    for name, module in model.named_modules():
        if hasattr(module, "store_attn_map"):
            del module.store_attn_map

    if hasattr(module, "attn_hook"):
        module.attn_hook.remove()
        del module.attn_hook


def register_hooks(model, detach=True):
    """
    Register hooks to the model to store attention maps.
    """
    # Register hooks for all attention layers
    model = register_attention_hook(model, hook_function, "attn")
    
    return model


def clear_attn_maps():
    """
    Clear the stored attention maps.
    """
    attn_maps.clear()


def save_attention_image(attn_map, n_frames, height, width, root_dir, layer_name=None, timestep=None):
    attn_map = rearrange(attn_map, 'b (f h w) ... -> b f h w ...', f=n_frames, h=height, w=width)
    attn_map = torch.mean(attn_map, dim=(2,3))
    attn_map = torch.mean(attn_map, dim=(3,4)) # (batch, frame, frame)
    attn_map = attn_map.permute(0, 2, 1) # (batch, frame, frame)
    # normalize
    attn_map = attn_map / attn_map.max(dim=-1, keepdim=True)[0]

    for i, a in enumerate(attn_map):
        batch_dir = os.path.join(root_dir, f'sample-{i}')
        if timestep is not None:
            batch_dir = os.path.join(batch_dir, f'timestep-{timestep}')
        os.makedirs(batch_dir, exist_ok=True)

        if layer_name:
            file_path = os.path.join(batch_dir, f'{layer_name}.png')
        else:
            file_path = os.path.join(batch_dir, f'attn_map.png')

        a = F.normalize(a, dim=1)

        heat_map = a.cpu().numpy()
        # plot heatmap
        plt.imshow(heat_map, cmap='viridis', interpolation='nearest')
        # x title: query, y title: key
        plt.xlabel('Key', fontsize=12)
        plt.ylabel('Query', fontsize=12)
        plt.colorbar()
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def save_attention_maps(attn_maps, base_dir='attn_maps', unconditional=False, batch_idx=0):
    to_pil = ToPILImage()
    base_dir = os.path.join(base_dir, f'batch-{batch_idx}')
    os.makedirs(base_dir, exist_ok=True)
    
    total_attn_map = list(list(attn_maps.values())[0].values())[0].sum(1)
    if unconditional:
        total_attn_map = total_attn_map.chunk(2)[1]

    total_attn_map = total_attn_map.permute(0, 4, 1, 2, 3) # (batch_size, dim, frames, height, width)
    batch_size, dim, frames, height, width = total_attn_map.shape
    n_tokens = frames * height * width
    tokens_per_frame = height * width

    total_attn_map = torch.zeros_like(total_attn_map)
    total_attn_map_shape = total_attn_map.shape[-2:]
    total_attn_map_number = 0
    
    for timestep, layers in attn_maps.items():       
        for layer, attn_map in layers.items():           
            attn_map = attn_map.sum(1).squeeze(1).permute(0, 4, 1, 2, 3)
            if unconditional:
                attn_map = attn_map.chunk(2)[1]
            
            attn_map = rearrange(attn_map, 'b dim frames height width -> b (dim frames) height width')
            resized_attn_map = F.interpolate(attn_map, size=total_attn_map_shape, mode='bilinear', align_corners=False)
            attn_map = rearrange(attn_map, 'b (dim frames) height width -> b dim frames height width', frames=frames)
            resized_attn_map = rearrange(resized_attn_map, 'b (dim frames) height width -> b dim frames height width', frames=frames)

            total_attn_map += resized_attn_map
            total_attn_map_number += 1

            save_attention_image(attn_map, frames, height, width, base_dir, layer, timestep)
    
    total_attn_map /= total_attn_map_number
    save_attention_image(total_attn_map, frames, height, width, base_dir)
    
    clear_attn_maps()