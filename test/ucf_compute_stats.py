import os
import numpy as np
import cv2
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
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return None
    
    # Initialize lists to store channel-wise data
    r_channel = []
    g_channel = []
    b_channel = []
    
    # Read frames
    while True:
        ret, frame = cap.read()
        
        # Break the loop if no more frames
        if not ret:
            break
        
        # Split the frame into color channels
        b, g, r = cv2.split(frame)
        
        # Append channel data (flatten to 1D array)
        r_channel.append(r.flatten())
        g_channel.append(g.flatten())
        b_channel.append(b.flatten())
    
    # Release the video capture object
    cap.release()
    
    # If no frames were read
    if not r_channel:
        print(f"No frames found in video: {video_path}")
        return None
    
    # Combine all frames for each channel
    return (
        np.concatenate(r_channel),
        np.concatenate(g_channel),
        np.concatenate(b_channel)
    )

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
            if filename.lower().endswith('.mp4'):
                full_path = os.path.join(path, filename)
                video_files.append(full_path)
    
    # Use multiprocessing to compute stats
    with mp.Pool() as pool:
        # Use list to consume all results
        channel_data = list(pool.imap(compute_video_channel_stats, video_files))
    
    # Filter out None values
    channel_data = [data for data in channel_data if data is not None]
    
    # If no valid videos were found
    if not channel_data:
        print("No valid videos found!")
        return None
    
    # Aggregate channel data
    r_data = np.concatenate([video[0] for video in channel_data])
    g_data = np.concatenate([video[1] for video in channel_data])
    b_data = np.concatenate([video[2] for video in channel_data])
    
    # Compute aggregate statistics
    stats = {
        'red': {
            'mean': np.mean(r_data),
            'std': np.std(r_data)
        },
        'green': {
            'mean': np.mean(g_data),
            'std': np.std(g_data)
        },
        'blue': {
            'mean': np.mean(b_data),
            'std': np.std(b_data)
        }
    }
    
    return stats

def main():
    # Specify the folder containing MP4 videos
    video_folder = "/vast/s224075134/temporal_diffusion/FAR/datasets/ucf101/preprocessed_64_mp4"
    
    # Compute aggregate statistics
    aggregate_stats = aggregate_video_stats(video_folder)
    
    # Print results
    if aggregate_stats:
        print("Aggregate Video Channel Statistics:")
        for channel, stats in aggregate_stats.items():
            print(f"{channel.capitalize()} Channel:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
        
        # Optionally save to a file
        import json
        with open('aggregate_video_stats.json', 'w') as f:
            json.dump(aggregate_stats, f, indent=2)

if __name__ == "__main__":
    main()