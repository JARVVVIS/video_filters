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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--clip_name",
        type=str,
        help="Name of clip whose frame(s) to test on.",
        default="trimmed_chameleon",
    )
    args = parser.parse_args()

    video_path = "/fs/nexus-projects/mt_sec/t2v_curate/assets/split_scenes/(500)_Days_of_Summer_(2_5)_Movie_CLIP_-_Playing_House_(2009)_HD/scene_001.mp4"
    frames = return_frames(clip_path=video_path)

    artifacts = load_artifacts(args)
    results = infer(frames_list=frames, artifacts=artifacts)
    print(f"Results: {results['motion_fb']}")
