import os
import json
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from download_yt import download_video

from video_filters.frame_utils import extract_frames_with_timestamp
from configs import ROOT_DIR, FILTER_MODULES, get_filter_module


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
    video_id, video_path, dataset_name, discard_last=None, needs_video=False
):
    """Process a single video: download if needed, extract frames, and return frame paths"""
    downloaded_tmp = False

    # If video_path is a URL, download it
    if needs_video and (not os.path.exists(video_path)):
        # Must be YT-URL; download the video with a tmp-name
        tmp_video_path = f"{ROOT_DIR}/assets/clips/tmp_{video_id}"
        video_path, did_download = download_video(
            video_path, tmp_video_path, root=f"{ROOT_DIR}/assets/clips"
        )
        assert did_download
        assert os.path.exists(video_path)
        print(f"Downloaded video @ {video_path}")
        downloaded_tmp = True

    # Extract frames if they don't exist
    frames_path = f"{ROOT_DIR}/assets/extracted_frames_{dataset_name}/{video_id}"

    ## check files inside the directory
    ## if the directory is empty, extract frames
    num_frames_present = (
        len(os.listdir(frames_path)) if os.path.exists(frames_path) else 0
    )

    if not num_frames_present:
        os.makedirs(frames_path, exist_ok=True)
        extract_frames_with_timestamp(
            video_path, frames_path, discard_last=discard_last
        )

    # Get all image paths sorted
    image_paths = [
        os.path.join(frames_path, image_path)
        for image_path in sorted(os.listdir(frames_path))
    ]
    assert len(image_paths) > 0, f"No frames found in {frames_path}"
    print(f"Extracted {len(image_paths)} frames from {video_path}")

    return image_paths, video_path, downloaded_tmp


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

    args = parser.parse_args()

    assert [
        filter in FILTER_MODULES for filter in args.filters
    ], f"Invalid filters: {args.filters}"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get video paths for the dataset
    video_dict, original_df = eval(f"get_{args.dataset}_paths(args.dataset)")

    # Process each filter (outer loop)
    for filter_name in args.filters:
        print(f"Processing filter: {filter_name}")

        # Initialize filter
        filter_module, needs_video = get_filter_module(filter_name)
        filter_artifacts = filter_module.load_artifacts(args)

        # Initialize an empty DataFrame for this filter's results
        filter_results_dicts = []

        count = 0
        # Process each video (inner loop)
        for video_id, video_path in video_dict.items():
            count += 1
            if count == 10:
                break

            print(f"  Processing video: {video_id}")

            # Process the video: download if needed, extract frames
            image_paths, actual_video_path, is_tmp = process_video(
                video_id, video_path, args.dataset, args.discard_last, needs_video
            )

            # Run the filter on this video's frames
            filter_results = filter_module.infer(
                image_paths=image_paths, artifacts=filter_artifacts
            )

            ## update the columns with the results
            update_dict = {"video_id": video_id, **filter_results}
            filter_results_dicts.append(update_dict)

            # Clean up temporary downloaded video to save memory
            if is_tmp and os.path.exists(actual_video_path):
                print(f"  Removing temporary video: {actual_video_path}")
                os.remove(actual_video_path)

        # Convert the list of dictionaries to a DataFrame
        filter_results_df = pd.DataFrame(
            filter_results_dicts, columns=["video_id"] + list(filter_results.keys())
        )

        # Save the results for this filter
        output_path = os.path.join(
            args.output_dir, f"{filter_name}_results_{args.dataset}"
        )
        filter_results_df.to_csv(f"{output_path}.csv", index=False)
        filter_results_df.to_pickle(f"{output_path}.pkl")
        print(f"Saved results for {filter_name} to {output_path}")


if __name__ == "__main__":
    main()
