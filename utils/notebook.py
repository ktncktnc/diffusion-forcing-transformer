import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import torch


def show_video(video: torch.Tensor):
    """
    Show a video in a Jupyter notebook
    Args:
        video: A tensor of shape (T, C, H, W) or (T, H, W, C)
    """
    rc('animation', html='jshtml')
    if len(video.shape) == 4:
        video = video.permute(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
    elif len(video.shape) != 5:
        raise ValueError("Video must be of shape (T, C, H, W) or (T, H, W, C)")
    
    fig, ax = plt.subplots()
    imgs = torch.clamp(video.cpu(), 0, 1)
    frames = [[ax.imshow(imgs[i])] for i in range(len(imgs))]
    ani = animation.ArtistAnimation(fig, frames)
    return ani