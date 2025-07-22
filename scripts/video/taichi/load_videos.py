import numpy as np
import pandas as pd
import imageio
import os
import subprocess
from multiprocessing import Pool
from itertools import cycle
import warnings
import glob
import time
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
from yt_dlp import YoutubeDL
warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')


def save(path, frames, format):
    # print("Saving %s" % path)
    if format == '.mp4':
        if os.path.exists(path):
            print("Warning: skipping video %s" % os.path.basename(path))
            return
        imageio.mimsave(path, frames)
    elif format == '.png':
        if os.path.exists(path):
            print ("Warning: skiping video %s" % os.path.basename(path))
            return
        else:
            os.makedirs(path)
        for j, frame in enumerate(frames):
            imageio.imsave(os.path.join(path, str(j).zfill(7) + '.png'), frames[j]) 
    else:
        print ("Unknown format %s" % format)
        exit()


def download(video_id, args):
    # print("Downloading video %s" % video_id)
    video_path = os.path.join(args.video_folder, video_id + ".mp4")
    subprocess.run([args.youtube, '-f', "''best/mp4''", '--write-auto-sub', '--write-sub',
                    '--sub-lang', 'en', '--skip-unavailable-fragments',
                    "https://www.youtube.com/watch?v=" + video_id, "--output", video_path], stdout=DEVNULL)
    
    return video_path

def run(data):
    video_id, args = data

    # Get metadata for the video
    df = pd.read_csv(args.metadata)
    df = df[df['video_id'] == video_id]
    ref_fps = df['fps'].iloc[0]
    ref_height = df['height'].iloc[0]
    ref_width = df['width'].iloc[0]
    partition = df['partition'].iloc[0]
    
    # Check if the video is already processed
    all_chunks_dict = []
    all_done = True
    for j in range(df.shape[0]):
        start = df['start'].iloc[j]
        end = df['end'].iloc[j]
        bbox = list(map(int, df['bbox'].iloc[j].split('-')))
        frames = []
        path = os.path.join(args.out_folder, partition, '#'.join(video_id.split('#')[::-1]) + '#' + str(start).zfill(6) + '#' + str(end).zfill(6) + '.mp4')
        all_chunks_dict.append({
            'start': start, 
            'end': end, 
            'bbox': bbox, 
            'frames': frames,
            'path': path
        })
        if not os.path.exists(path):
            all_done = False
    
    # Check if all chunks are already processed
    if all_done:
        # print('All chunks for video %s already processed' % video_id)
        return
    
    # Download the video if it does not exist
    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):
        try:
            download(video_id.split('#')[0], args)
        except Exception as e:
            print('Error downloading video %s: %s' % (video_id.split('#')[0], e))
            return
    else:
        #print('Video %s already exists' % video_id.split('#')[0])
        pass

    # Check if the video file is valid
    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):
        # print ('Can not load video %s, broken link' % video_id.split('#')[0])
        return
    
    # Read video
    reader = imageio.get_reader(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4'))
    fps = reader.get_meta_data()['fps']

    # print('Processing video %s' % (video_id))
    try:
        for i, frame in enumerate(reader):
            for entry in all_chunks_dict:
                if (i * ref_fps >= entry['start'] * fps) and (i * ref_fps < entry['end'] * fps):
                    left, top, right, bot = entry['bbox']
                    left = int(left / (ref_width / frame.shape[1]))
                    top = int(top / (ref_height / frame.shape[0]))
                    right = int(right / (ref_width / frame.shape[1]))
                    bot = int(bot / (ref_height / frame.shape[0]))
                    crop = frame[top:bot, left:right]

                    if args.image_shape is not None:
                        crop = img_as_ubyte(resize(crop, args.image_shape, anti_aliasing=True))

                    entry['frames'].append(crop)

    except Exception as e:
        print('Error', e)
        raise e

    for entry in all_chunks_dict:
        first_part = '#'.join(video_id.split('#')[::-1])
        path = first_part + '#' + str(entry['start']).zfill(6) + '#' + str(entry['end']).zfill(6) + '.mp4'
        save(os.path.join(args.out_folder, partition, path), entry['frames'], args.format)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_folder", default='youtube-taichi', help='Path to youtube videos')
    parser.add_argument("--metadata", default='metadata.csv', help='Path to metadata')
    parser.add_argument("--out_folder", default='taichi-png', help='Path to output')
    parser.add_argument("--format", default='.png', help='Storing format')
    parser.add_argument("--workers", default=4, type=int, help='Number of workers')
    parser.add_argument("--youtube", default='yt-dlp', help='Path to youtube-dl')
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))), help="Image shape, None for no resize")

    args = parser.parse_args()
    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for partition in ['test', 'train']:
        if not os.path.exists(os.path.join(args.out_folder, partition)):
            os.makedirs(os.path.join(args.out_folder, partition))

    df = pd.read_csv(args.metadata)
    video_ids = set(df['video_id'])
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for chunks_data in tqdm(pool.imap_unordered(run, zip(video_ids, args_list))):
        None
    # for chunks_data in tqdm(zip(video_ids, args_list)):
    #     run(chunks_data)