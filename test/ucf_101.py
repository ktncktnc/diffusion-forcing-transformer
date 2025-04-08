from datasets.video.ucf_101 import UCF101BaseVideoDataset
from datasets.video.ucf_101 import UCF101SimpleVideoDataset
from datasets.video.ucf_101 import UCF101AdvancedVideoDataset


from omegaconf import DictConfig
import yaml


with open('/home/s224075134/diffusion-forcing-transformer/configurations/dataset/ucf_101.yaml') as f:
    cfg = yaml.safe_load(f)
cfg = DictConfig(cfg)
print(cfg)

dataset = UCF101AdvancedVideoDataset(cfg, split='training')
print(dataset)
item = dataset[1]

print(item['videos'].shape)
print(item['nonterminal'].shape)

# save video
import cv2
import numpy as np
video = item['videos'].numpy()
video = np.transpose(video, (0, 2, 3, 1))
video = (video * 255).astype(np.uint8)
video_path = '/home/s224075134/diffusion-forcing-transformer/test/video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
height, width = video.shape[1:3]
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
for frame in video:
    video_writer.write(frame)
video_writer.release()
# save nonterminal