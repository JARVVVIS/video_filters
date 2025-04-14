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
from video_filters.frame_utils import get_key_frames


def load_artifacts(args):
    return {}


def infer(image_paths, artifacts):
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

    image_dir = os.path.join("../../assets/extracted_frames", args.clip_name)
    image_paths = [
        os.path.join(image_dir, image_path)
        for image_path in sorted(os.listdir(image_dir))
    ]
    artifacts = load_artifacts(args)
    results = infer(image_paths=image_paths, artifacts=artifacts)
    print(f"Results: {results['motion_scores']}")
