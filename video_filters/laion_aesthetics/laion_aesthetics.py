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
from video_filters.frame_utils import get_key_frames


def load_artifacts(args):
    aesthetic_model = load_aesthetic_laion(
        device=args.device, model_path=args.aesthetic_model_path, dtype=args.dtype
    )
    return {"aesthetic_model": aesthetic_model}


@torch.no_grad()
@torch.inference_mode()
def infer(image_paths, artifacts, sample_prop=1):
    aesthetic_model = artifacts["aesthetic_model"]

    aesthetic_results = {image_path: None for image_path in image_paths}

    frames = [Image.open(image_path) for image_path in image_paths]
    ## sample "sample_prop" proportion of frames uniformly from frames
    frames = frames[::sample_prop]
    scores = [
        tensor.cpu().item() for tensor in run_aesthetic_laion(aesthetic_model, frames)
    ]
    aesthetic_results = {
        image_path.split("/")[-1]: score
        for image_path, score in zip(image_paths, scores)
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

    image_dir = os.path.join("../../assets/extracted_frames", args.clip_name)
    image_paths = [
        os.path.join(image_dir, image_path)
        for image_path in sorted(os.listdir(image_dir))
    ]
    artifacts = load_artifacts(args)
    results = infer(image_paths=image_paths, artifacts=artifacts)
    print(f"Result for one frame: {results['aesthetic_scores'][image_paths[0]]}")
