import time
import pandas as pd
import pathlib
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
import os
from video_filters.motion_score.utils import (
    compute_farneback_optical_flow,
    compute_lk_optical_flow,
    _downscale_maps,
    _motion_score,
)
from video_filters.frame_utils import return_frames


def load_artifacts(args):
    return {}


def infer(image_paths=None, artifacts=None, frames_list=None):
    if image_paths is None and frames_list is None:
        raise ValueError("Either image_paths or frames_list must be provided.")
    if frames_list is not None:
        frames = frames_list
    else:
        frames = [Image.open(image_path) for image_path in image_paths]

    farneback, _, _, _ = compute_farneback_optical_flow(frames)
    farneback = _motion_score(_downscale_maps(farneback))
    lucas_kanade = _motion_score(compute_lk_optical_flow(frames))
    results = {"motion_fb": farneback, "motion_lk": lucas_kanade}

    return results

def infer_batch(image_paths=None, artifacts=None, frames_list=None):
    outputs = []
    for fr in frames_list:
        results = infer(image_paths=image_paths, artifacts=artifacts, frames_list=fr)
        outputs.append(results)
    return outputs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--clip_name",
        type=str,
        help="Name of clip whose frame(s) to test on.",
        default="trimmed_chameleon",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for inference.",
        default=24,
    )
    args = parser.parse_args()

    video_path = "/fs/nexus-projects/mt_sec/t2v_curate/assets/split_scenes_long/(500)_Days_of_Summer_(2_5)_Movie_CLIP_-_Playing_House_(2009)_HD/scene_003.mp4"
    assert os.path.exists(video_path), f"Video path {video_path} does not exist."
    frames = return_frames(clip_path=video_path)

    frames_list = [frames] * args.batch_size
    artifacts = load_artifacts(args)

    seq_start_time = time.time()
    sequential_results = []
    for frame_list_idx, fr in enumerate(frames_list):
        res = infer(frames_list=fr, artifacts=artifacts)
        sequential_results.append(res)
        print(
            f"[SEQ {frame_list_idx}] motion_fb: {res['motion_fb']:.6f} | motion_lk: {res['motion_lk']:.6f}"
        )
    seq_end_time = time.time()
    print(f"Sequential inference took {seq_end_time - seq_start_time:.2f} seconds.")

    batch_start_time = time.time()
    batched_results = infer_batch(frames_list=frames_list, artifacts=artifacts)
    batch_end_time = time.time()
    print(f"Batched inference took {batch_end_time - batch_start_time:.2f} seconds.")

    for frame_list_idx, res in enumerate(batched_results):
        print(
            f"[BATCH {frame_list_idx}] motion_fb: {res['motion_fb']:.6f} | motion_lk: {res['motion_lk']:.6f}"
        )
