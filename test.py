import torch

def generate_gibbs_attention_mask(n_frames: int, n_tokens_per_frame: int, frame_idx) -> torch.Tensor:
    """
    Generate a Gibbs attention mask for the given number of frames and tokens per frame.
    Args:
        n_frames: Number of frames.
        n_tokens_per_frame: Number of tokens per frame. In case of factorized attention, n_tokens_per_frame = 1
    Returns:
        A tensor representing the attention mask.
    """
    # Create a mask with shape (n_frames, n_tokens_per_frame, n_tokens_per_frame)
    mask = torch.ones((n_frames*n_tokens_per_frame, n_frames*n_tokens_per_frame), device='cpu')
    mask[frame_idx*n_tokens_per_frame:(frame_idx+1)*n_tokens_per_frame, frame_idx*n_tokens_per_frame:(frame_idx+1)*n_tokens_per_frame] = 0
    return mask

mask = generate_gibbs_attention_mask(4, 2, 1)
print(mask)