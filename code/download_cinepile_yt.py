"""
Modified version of your workflow with integrated watermark cropping and parallelization support
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from download_yt import download_video
from video_filters.frame_utils import extract_frames_with_timestamp
# Import the simple watermark cropper and video trimmer
from video_filters.cropper import simple_crop_video_watermarks, SimpleWatermarkCropper
from video_filters.trim import trim_video_end, get_video_info

ROOT_DIR="/Volumes/T7 Backup/video_filters"

def get_cinepile_paths(dataset):
    """Load all frames/video paths for cinepile dataset"""
    cinepile_df = load_dataset(
        "tomg-group-umd/cinepile", "v2", split="train"
    ).to_pandas()
    # Create a dict mapping unique titles to their links
    unique_titles = cinepile_df["yt_clip_title"].unique()
    video_dict = {}
    for title in unique_titles:
        # Get the index of the title and its corresponding link
        link = cinepile_df[cinepile_df["yt_clip_title"] == title]["yt_clip_link"].iloc[
            0
        ]
        video_dict[title] = link
    return video_dict, cinepile_df

def process_video(
    video_id, video_path, dataset_name, discard_last=None, needs_video=False,
    enable_watermark_cropping=True, bottom_crop_percent=15.0, top_crop_percent=10.0,
    use_simple_crop=False, enable_end_trimming=True
):
    """Process a single video: download if needed, crop watermarks, extract frames, and return frame paths"""
    downloaded_tmp = False
    # Extract frames if they don't exist
    frames_path = f"{ROOT_DIR}/assets/extracted_frames_{dataset_name}/{video_id}"
    ## check files inside the directory
    ## if the directory is empty, extract frames
    if os.path.exists(frames_path):
        num_frames_present = (
            len(os.listdir(frames_path)) if os.path.exists(frames_path) else 0
        )
    else:
        num_frames_present = 0
    
    tmp_video_path = f"{ROOT_DIR}/assets/clips/{video_id}"
    if os.path.exists(tmp_video_path + '.mp4'):
        print(f"Using existing video @ {tmp_video_path}.mp4")
        return None, tmp_video_path, True
    
    video_path, did_download = download_video(
        video_path, tmp_video_path, root=f"{ROOT_DIR}/assets/clips"
    )
    if not did_download:
        return None, None, False
    assert os.path.exists(video_path)
    print(f"Downloaded video @ {video_path}")
    downloaded_tmp = True
    
    # NEW: Trim end of video if requested (before watermark cropping)
    if enable_end_trimming and discard_last and discard_last > 0:
        print(f"Trimming last {discard_last} seconds from video: {video_path}")
        try:
            trimming_success = trim_video_end(video_path, discard_last)
            if trimming_success:
                print(f"Successfully trimmed {discard_last}s from {video_path}")
            else:
                print(f"Failed to trim video {video_path}")
        except Exception as e:
            print(f"Error during video trimming for {video_path}: {str(e)}")
            print("Continuing with original video...")
    
    # NEW: Preprocess video to remove watermarks before extracting frames
    if enable_watermark_cropping:
        print(f"Preprocessing video for watermark removal: {video_path}")
        try:
            # Use smart detection by default, crops both top and bottom black regions
            cropping_success = simple_crop_video_watermarks(
                video_path, 
                bottom_percent=bottom_crop_percent,
                top_percent=top_crop_percent,
                use_smart_detection=not use_simple_crop
            )
            if cropping_success:
                print(f"Successfully processed watermarks for {video_path}")
            else:
                print(f"Watermark processing completed (no watermarks detected or error occurred) for {video_path}")
        except Exception as e:
            print(f"Error during watermark processing for {video_path}: {str(e)}")
            print("Continuing with original video...")
    
    return None, video_path, downloaded_tmp

def batch_preprocess_existing_videos(video_folder_path: str, file_pattern: str = "*.mp4",
                                   bottom_percent: float = 15.0, top_percent: float = 10.0,
                                   discard_last: int = None):
    """
    Batch preprocess existing videos in a folder to remove watermarks and trim ends.
    
    Args:
        video_folder_path: Path to folder containing videos
        file_pattern: File pattern to match (default: "*.mp4")
        bottom_percent: Percentage of bottom to check for black regions
        top_percent: Percentage of top to check for black regions
        discard_last: Seconds to trim from end of each video (optional)
    """
    import glob
    
    video_files = glob.glob(os.path.join(video_folder_path, file_pattern))
    print(f"Found {len(video_files)} video files to preprocess")
    
    successful = 0
    failed = 0
    
    for video_file in video_files:
        print(f"Processing: {os.path.basename(video_file)}")
        try:
            # First trim the end if requested
            if discard_last and discard_last > 0:
                print(f"  Trimming last {discard_last} seconds...")
                trim_success = trim_video_end(video_file, discard_last)
                if not trim_success:
                    print(f"  Failed to trim {os.path.basename(video_file)}")
                    failed += 1
                    continue
            
            # Then remove watermarks
            print(f"  Removing watermarks...")
            success = simple_crop_video_watermarks(
                video_file,
                bottom_percent=bottom_percent,
                top_percent=top_percent,
                use_smart_detection=True
            )
            if success:
                successful += 1
                print(f"Successfully processed: {os.path.basename(video_file)}")
            else:
                failed += 1
                print(f"Failed to process: {os.path.basename(video_file)}")
        except Exception as e:
            failed += 1
            print(f"Error processing {os.path.basename(video_file)}: {str(e)}")
    
    print(f"\nBatch processing complete:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(video_files)}")

def get_video_chunk(video_dict, start_idx=None, end_idx=None):
    """
    Get a chunk of videos from the video dictionary based on start and end indices.
    
    Args:
        video_dict: Dictionary of video_id -> video_path
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive)
    
    Returns:
        Dictionary containing the specified chunk of videos
    """
    video_items = list(video_dict.items())
    total_videos = len(video_items)
    
    # Handle default values
    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = total_videos
    
    # Validate indices
    start_idx = max(0, start_idx)
    end_idx = min(total_videos, end_idx)
    
    if start_idx >= end_idx:
        print(f"Warning: start_idx ({start_idx}) >= end_idx ({end_idx}). No videos to process.")
        return {}
    
    chunk_items = video_items[start_idx:end_idx]
    chunk_dict = dict(chunk_items)
    
    print(f"Processing videos {start_idx} to {end_idx-1} ({len(chunk_dict)} videos out of {total_videos} total)")
    return chunk_dict

def print_dataset_info(video_dict, dataset_name):
    """Print information about the dataset for planning parallel processing"""
    print(f"\n=== Dataset Info: {dataset_name} ===")
    print(f"Total videos: {len(video_dict)}")
    print(f"Video IDs (first 5): {list(video_dict.keys())[:5]}")
    if len(video_dict) > 5:
        print(f"... and {len(video_dict) - 5} more")
    print("="*50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cinepile")
    parser.add_argument("--output_dir", type=str, default=f"{ROOT_DIR}/captions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--discard_last", type=int, default=25)
    parser.add_argument(
        "--caption_strategies", nargs="+", default=["base", "scene_text"]
    )
    parser.add_argument("--model_name", type=str, default="qwen2_5_vl_7b_instruct")
    
    # NEW: Parallelization arguments
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Starting index for video processing (inclusive)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Ending index for video processing (exclusive)")
    parser.add_argument("--info_only", action="store_true", default=False,
                        help="Only print dataset info and exit (useful for planning parallel jobs)")
    
    # NEW: Add watermark cropping and video trimming options
    parser.add_argument("--enable_watermark_cropping", action="store_true", default=True,
                        help="Enable watermark detection and cropping")
    parser.add_argument("--enable_end_trimming", action="store_true", default=True,
                        help="Enable trimming of video end")
    parser.add_argument("--batch_preprocess_folder", type=str, default=None,
                        help="Batch preprocess existing videos in this folder")
    parser.add_argument("--bottom_crop_percent", type=float, default=15.0,
                        help="Percentage of bottom to check for watermarks")
    parser.add_argument("--top_crop_percent", type=float, default=10.0,
                        help="Percentage of top to check for watermarks")
    parser.add_argument("--use_simple_crop", action="store_true", default=False,
                        help="Use simple percentage cropping instead of smart detection")
    
    args = parser.parse_args()
    
    # NEW: If batch preprocessing is requested, do that and exit
    if args.batch_preprocess_folder:
        if os.path.exists(args.batch_preprocess_folder):
            batch_preprocess_existing_videos(
                args.batch_preprocess_folder,
                bottom_percent=args.bottom_crop_percent,
                top_percent=args.top_crop_percent,
                discard_last=args.discard_last
            )
        else:
            print(f"Folder not found: {args.batch_preprocess_folder}")
        return
    
    args.output_dir = f"{args.output_dir}/{args.dataset}"
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get video paths for the dataset
    video_dict, original_df = eval(f"get_{args.dataset}_paths(args.dataset)")
    
    # NEW: Print dataset info if requested
    if args.info_only:
        print_dataset_info(video_dict, args.dataset)
        return
    
    # NEW: Get the chunk of videos to process
    video_chunk = get_video_chunk(video_dict, args.start_idx, args.end_idx)
    
    if not video_chunk:
        print("No videos to process. Exiting.")
        return
    
    # Create a set of videos already processed for each strategy
    existing_videos = {}
    count = 0
    successful_count = 0
    failed_count = 0
    
    # Process each video (outer loop) - now only processing the chunk
    for video_id, video_path in video_chunk.items():
        count += 1
        print(f"\n[{count}/{len(video_chunk)}] Processing video: {video_id}")
        
        try:
            # Get the row from the original dataframe for this video
            video_row = original_df[original_df["yt_clip_title"] == video_id].iloc[0]
            movie_scene_text = video_row["movie_scene"]
            
            video_id_clean = video_id.replace(" ", "_").replace("/", "_").replace(":", "_").replace(".", "_")
            
            # Process the video with watermark cropping and end trimming enabled
            image_paths, actual_video_path, is_tmp = process_video(
                video_id_clean, video_path, args.dataset, args.discard_last, True,
                enable_watermark_cropping=args.enable_watermark_cropping,
                bottom_crop_percent=args.bottom_crop_percent,
                top_crop_percent=args.top_crop_percent,
                use_simple_crop=args.use_simple_crop,
                enable_end_trimming=args.enable_end_trimming
            )
            
            if actual_video_path:
                successful_count += 1
                print(f"✓ Successfully processed: {video_id}")
            else:
                failed_count += 1
                print(f"✗ Failed to process: {video_id}")
                
        except Exception as e:
            failed_count += 1
            print(f"✗ Error processing {video_id}: {str(e)}")
    
    # Summary
    print(f"\n=== Processing Summary ===")
    print(f"Processed chunk: indices {args.start_idx or 0} to {args.end_idx or len(video_dict)}")
    print(f"Videos processed: {count}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {failed_count}")
    print("="*25)

if __name__ == "__main__":
    main()