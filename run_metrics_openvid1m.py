from pathlib import Path
import os
import argparse
import pandas as pd

from video_filters.frame_utils import return_frames
from configs import ROOT_DIR, FILTER_MODULES, get_filter_module


def get_openvid_paths(root_dir):
    if isinstance(root_dir, str):
        root_dir = Path(str)
    video_paths = list(root_dir.glob("*.mp4"))
    return sorted(video_paths)

def process_video(video_path, discard_last=None):
    frames_list = return_frames(video_path, discard_last=discard_last)
    return frames_list, video_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cinepile")
    parser.add_argument("--filters", nargs="+", default=["laion_aesthetics"])
    parser.add_argument("--output_dir", type=str, default=f"{ROOT_DIR}/results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--aesthetic_model_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--discard_last", type=int, default=None)
    parser.add_argument("--watermark_model_path", type=str, default=None)
    parser.add_argument(
        "--shot_categorizer_repo_id", type=str, default="diffusers/shot-categorizer-v0"
    )
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)

    args = parser.parse_args()

    for f in args.filters:
        if f not in FILTER_MODULES:
            raise ValueError(f"Invalid filter: {f}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get video paths for the dataset
    video_paths = get_openvid_paths(args.root_dir)

    # Convert video_dict to list for indexing
    total_videos = len(video_paths)

    # Handle start_idx and end_idx
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = total_videos

    if args.start_idx < 0 or args.end_idx > total_videos:
        args.start_idx = max(0, args.start_idx)
        args.end_idx = min(total_videos, args.end_idx)

    video_items_subset = video_paths[args.start_idx:args.end_idx]

    # scenes_dir_parent = f'{ROOT_DIR}/assets/split_scenes' ## TODO: UPDATE

    #### Long a bunch of stuff
    print('*'*100)
    print(f'Saving Results to: {args.output_dir}')
    print(f'Loading Videos from: {args.dataset}')
    # print(f'Processing Videos: {args.start_idx} to {args.end_idx-1} ({len(video_items_subset)} videos)')
    print(f'Filters: {args.filters}')
    print('*'*100)

    # Process each filter (outer loop)
    for filter_name in args.filters:
        print(f"Processing filter: {filter_name}")

        args.filter_prefix = f"{ROOT_DIR}/video_filters/{filter_name}"

        # Initialize filter
        filter_module = get_filter_module(filter_name)
        filter_artifacts = filter_module.load_artifacts(args)

        # Define final output paths
        base_output_path = os.path.join(
            args.output_dir, f"{filter_name}_results_{args.dataset.replace('/', '_')}"
        )
        range_suffix = f"_{args.start_idx}_{args.end_idx}"
        output_path = base_output_path + range_suffix
        checkpoint_csv = f"{output_path}_checkpoint.csv"
        checkpoint_pkl = f"{output_path}_checkpoint.pkl"

        # If checkpoint exists, load it
        if os.path.exists(checkpoint_csv):
            print(f"Found existing checkpoint for '{filter_name}': {checkpoint_csv}")
            filter_results_df = pd.read_csv(checkpoint_csv)
        else:
            # Will be initialized once we see the first filter_results
            filter_results_df = pd.DataFrame()

        # Keep track of which videos have already been processed
        processed_video_ids = (
            set(filter_results_df["video_id"].unique())
            if not filter_results_df.empty
            else set()
        )
        
        count = 0
        for _, video_path in enumerate(video_items_subset, start=args.start_idx):
            video_id = str(video_path).split("/")[-1]
            if video_id in processed_video_ids:
                    print(f"Skipping already processed video: {video_id}")
                    continue
            if not os.path.exists(video_path):
                print(f' Video {video_id} not found at {video_path}. Skipping...')
                continue

            frames_list, _ = process_video(video_path, False)

            # Run the filter
            try:
                if filter_name == "vision_reward":
                    filter_results = filter_module.infer(
                        artifacts=filter_artifacts, video_source=video_path
                    )
                else:
                    filter_results = filter_module.infer(
                        frames_list=frames_list, artifacts=filter_artifacts
                    )
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                continue

            print(f"Processing video {count}/{len(video_paths)}: {video_id}")

            # Prepare a row for the checkpoint
            update_dict = {"video_id": f"{video_id}"}
            print(f"Filter results for {video_id}: {filter_results}")
            update_dict.update(filter_results)

            # If our main DataFrame is still empty, initialize columns
            if filter_results_df.empty:
                # The columns are assumed consistent for each filter
                columns = ["video_id"] + list(filter_results.keys())
                filter_results_df = pd.DataFrame(columns=columns)

            # Append new result
            temp_df = pd.DataFrame([update_dict], columns=filter_results_df.columns)
            filter_results_df = pd.concat(
                [filter_results_df, temp_df], ignore_index=True
            )
            processed_video_ids.add(video_id)

            # Write checkpoint after each video
            if count % 50 == 0:
                filter_results_df.to_csv(checkpoint_csv, index=False)
                filter_results_df.to_pickle(checkpoint_pkl)

        # After processing all videos for this filter, do a final save
        # Ensure "video_id" is the first column
        final_cols = list(filter_results_df.columns)
        if "video_id" in final_cols:
            final_cols.remove("video_id")
            final_cols = ["video_id"] + final_cols
        filter_results_df = filter_results_df.reindex(columns=final_cols)

        # Save final CSV + Pickle
        filter_results_df.to_csv(f"{output_path}.csv", index=False)
        filter_results_df.to_pickle(f"{output_path}.pkl")
        print(f"Saved final results for '{filter_name}' in {output_path}.csv / .pkl")


if __name__ == "__main__":
    main()
