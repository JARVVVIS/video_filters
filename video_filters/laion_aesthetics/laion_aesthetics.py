import os
import pandas as pd
import torch
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from video_filters.laion_aesthetics.utils import (
    load_aesthetic_laion,
    run_aesthetic_laion,
)
from video_filters.frame_utils import return_frames


def load_artifacts(args):
    aesthetic_model = load_aesthetic_laion(
        device=args.device, model_path=args.aesthetic_model_path, dtype=args.dtype
    )
    return {"aesthetic_model": aesthetic_model}


@torch.no_grad()
@torch.inference_mode()
def infer(image_paths=None, artifacts=None, frames_list=None, sample_prop=1):
    aesthetic_model = artifacts["aesthetic_model"]

    if image_paths is None and frames_list is None:
        raise ValueError("Either image_paths or frames_list must be provided.")
    if frames_list is not None:
        frames = frames_list
    else:
        frames = [Image.open(image_path) for image_path in image_paths]

    aesthetic_results = {frame_idx: None for frame_idx in (range(len(frames)))}
    ## sample "sample_prop" proportion of frames uniformly from frames
    frames = frames[::sample_prop]
    scores = [
        tensor.cpu().item() for tensor in run_aesthetic_laion(aesthetic_model, frames)
    ]
    aesthetic_results = {
        frame_idx: score
        for frame_idx, score in zip(range(len(frames)), scores)
    }

    avg_aesthetic_score = sum(scores) / len(scores)

    return {
        "avg_aesthetic_score": avg_aesthetic_score,
        "frame_aesthetic_scores": aesthetic_results,
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--clip_name",
        type=str,
        help="Name of clip whose frame(s) to test on.",
        default="trimmed_chameleon",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--aesthetic_model_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    video_path = "/fs/nexus-projects/mt_sec/t2v_curate/assets/split_scenes/(500)_Days_of_Summer_(2_5)_Movie_CLIP_-_Playing_House_(2009)_HD/scene_001.mp4"
    frames = return_frames(clip_path=video_path)
    
    artifacts = load_artifacts(args)
    results = infer(frames_list=frames, artifacts=artifacts)
    print(f'Average Aesthetic Score: {results["avg_aesthetic_score"]}')
