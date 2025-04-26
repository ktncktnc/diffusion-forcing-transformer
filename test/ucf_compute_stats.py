import os
import numpy as np
import cv2
import torch
import multiprocessing as mp

def compute_video_channel_stats(video_path):
    """
    Compute channel-wise statistics for a single video.
    
    Parameters:
    -----------
    video_path : str
        Path to the input MP4 video file
    
    Returns:
    --------
    tuple
        Tuple containing channel-wise pixel values (R, G, B)
    """
    # Open the video file
    latent = torch.load(video_path)
    channel = latent.shape[1]
    stats = latent.permute(1,0,2,3).reshape(channel, -1)
    # n_values = stats.shape[1]
    return stats

def aggregate_video_stats(video_folder):
    """
    Compute aggregate channel-wise statistics across multiple videos.
    
    Parameters:
    -----------
    video_folder : str
        Path to the folder containing MP4 videos
    
    Returns:
    --------
    dict
        Aggregate channel-wise mean and standard deviation
    """
    # Find all MP4 files

    video_files = []
    for path, subdirs, files in os.walk(video_folder):
        for filename in files:
            # Check if the file is an MP4
            if filename.lower().endswith('.pt'):
                full_path = os.path.join(path, filename)
                video_files.append(full_path)
    
    # Use multiprocessing to compute stats
    with mp.Pool() as pool:
        # Use list to consume all results
        channel_data = list(pool.imap(compute_video_channel_stats, video_files))
    
    # Filter out None values
    channel_data = torch.cat(channel_data, dim=1)
    mean = torch.mean(channel_data, dim=1).unsqueeze(1).unsqueeze(2)
    std = torch.std(channel_data, dim=1).unsqueeze(1).unsqueeze(2)
    # Create a dictionary to store the results
    aggregate_stats = {
        'mean': mean,
        'std': std
    }
    # Calculate mean and std for each channel
    print(aggregate_stats)
    



def main():
    # Specify the folder containing MP4 videos
    video_folder = "/weka/s224075134/temporal_diffusion/datasets/video/bair_latent_8_1a8547fb/"
    
    # Compute aggregate statistics
    aggregate_stats = aggregate_video_stats(video_folder)
    
    # Print results
    # if aggregate_stats:
    #     print("Aggregate Video Channel Statistics:")
    #     for channel, stats in aggregate_stats.items():
    #         print(f"{channel.capitalize()} Channel:")
    #         print(f"  Mean: {stats['mean']:.4f}")
    #         print(f"  Std:  {stats['std']:.4f}")
        
    #     # Optionally save to a file
    #     import json
    #     with open('aggregate_video_stats.json', 'w') as f:
    #         json.dump(aggregate_stats, f, indent=2)

if __name__ == "__main__":
    main()