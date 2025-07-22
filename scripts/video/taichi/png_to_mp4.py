import pandas as pd
import cv2
import os
import glob
from pathlib import Path
import time
from multiprocessing import Pool

def create_video_from_frames(frame_folder, output_path, fps=30):
    """
    Create MP4 video from numbered PNG frames in a folder
    
    Args:
        frame_folder (str): Path to folder containing frame_XXXX.png files
        output_path (str): Output video file path (should end with .mp4)
        fps (int): Frames per second for the output video
    """
    
    # Get all PNG files that match the frame pattern and sort them numerically
    png_files = glob.glob(os.path.join(frame_folder, "frame_*.png"))
    
    if not png_files:
        print(f"No frame PNG files found in {frame_folder}")
        return False
    
    # Sort files numerically by frame number
    png_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    print(f"Processing {len(png_files)} frames from {os.path.basename(frame_folder)}")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(png_files[0])
    if first_image is None:
        print(f"Could not read first image: {png_files[0]}")
        return False
        
    height, width, layers = first_image.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for i, png_file in enumerate(png_files):
        if i % 50 == 0:  # Progress update every 50 frames
            print(f"  Processing frame {i+1}/{len(png_files)}")
        
        # Read image
        img = cv2.imread(png_file)
        
        if img is None:
            print(f"Warning: Could not read image {png_file}")
            continue
        
        # Resize image if needed (ensure all images have same dimensions)
        # if img.shape[:2] != (height, width):
        #     img = cv2.resize(img, (width, height))
        
        # Write frame to video
        video_writer.write(img)
    
    # Release everything
    video_writer.release()
    
    print(f"Video saved as: {output_path}")
    return True, output_path

def process_single_subfolder(args):
    """
    Wrapper function for multiprocessing
    
    Args:
        args: tuple containing (subfolder_path, output_video_path, fps)
    
    Returns:
        tuple: (success: bool, folder_name: str, message: str)
    """
    subfolder_path, output_video_path, fps = args
    return create_video_from_frames(subfolder_path, output_video_path, fps)


def process_all_subfolders(parent_folder, metadata, output_folder=None):
    """
    Process all subfolders in parent folder and create MP4 for each
    
    Args:
        parent_folder (str): Path to parent folder containing subfolders
        output_folder (str): Output folder for MP4 files (defaults to parent_folder)
        fps (int): Frames per second for output videos
    """
    
    parent_path = Path(parent_folder)
    
    if not parent_path.exists():
        print(f"Parent folder does not exist: {parent_folder}")
        return
    
    # Use parent folder as output if not specified
    if output_folder is None:
        output_folder = parent_folder
    
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    df = pd.read_csv(metadata)
    
    # Get all subdirectories
    subfolders = [f for f in parent_path.iterdir() if f.is_dir()]
    
    if not subfolders:
        print(f"No subfolders found in {parent_folder}")
        return
    
    print(f"Found {len(subfolders)} subfolders to process")
    
    successful = 0
    failed = 0
    process_args = []
    
    for subfolder in subfolders:
        subfolder_name = subfolder.name
        print(f"\n--- Processing subfolder: {subfolder_name} ---")

        video_idx = subfolder_name.split('#')[0]
        video_df = df[df['video_id'] == video_idx]
        fps = video_df['fps'].iloc[0]
        # Create output video name from subfolder name
        # Replace problematic characters for filename
        # safe_name = subfolder_name.replace('#', '_').replace('/', '_').replace('\\', '_')
        output_video_path = output_path / f"{subfolder_name}.mp4"
        process_args.append((str(subfolder), str(output_video_path), fps))
        
        # Create video from this subfolder
        # if create_video_from_frames(str(subfolder), str(output_video_path), fps):
        #     successful += 1
        # else:
        #     failed += 1
    
    start_time = time.time()
    
    # Process in parallel
    with Pool(processes=6) as pool:
        results = pool.map(process_single_subfolder, process_args)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Process results
    successful = 0
    failed = 0
    
    print(f"\n=== Results ===")
    for success, folder_name in results:
        if success:
            successful += 1
        else:
            failed += 1
            print(f"✗ {folder_name} - Failed to create video")
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful} videos")
    print(f"Failed: {failed} videos")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per video: {elapsed_time/len(subfolders):.2f} seconds")
    print(f"Output folder: {output_folder}")


# Example usage
if __name__ == "__main__":
    parent_directory = "/scratch/s224075134/temporal_diffusion/datasets/video/hg_taichi/taichi-256/frames/train"
    output_folder = "/scratch/s224075134/temporal_diffusion/datasets/video/hg_taichi/taichi-256/videos/train"
    metadata = '/home/s224075134/diffusion-forcing-transformer/scripts/video/taichi/metadata.csv'
    process_all_subfolders(parent_directory, metadata, output_folder)

    # Method 1: Process ALL subfolders in a directory
    parent_directory = "/scratch/s224075134/temporal_diffusion/datasets/video/hg_taichi/taichi-256/frames/test"
    output_folder = "/scratch/s224075134/temporal_diffusion/datasets/video/hg_taichi/taichi-256/videos/test"
    metadata = '/home/s224075134/diffusion-forcing-transformer/scripts/video/taichi/metadata.csv'
    process_all_subfolders(parent_directory, metadata, output_folder)
    
    # Method 2: Process all subfolders and save videos to different location
    # process_all_subfolders("input_folder", "output_videos", fps=30)
    
    # Method 3: Process only specific subfolders
    # specific_folders = [
    #     "ixmigvyetj4/#001164#001293",
    #     "Q0tIcxMcm4E#006354#006594",
    #     "Yqt573H689c#000026#000451"
    # ]
    # process_specific_subfolders(parent_directory, specific_folders, fps=24)
    
    # Method 4: Process single subfolder
    # single_subfolder = "path/to/parent/ixmigvyetj4/#001164#001293"
    # create_video_from_frames(single_subfolder, "output_video.mp4", fps=30)