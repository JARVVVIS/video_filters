import os
import pandas as pd
import torch
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from video_filters.laion_nsfw.utils import load_nsfw, run_nsfw
from video_filters.frame_utils import get_key_frames


def load_artifacts(args):
    nsfw_model, nsfw_processor = load_nsfw(device=args.device)
    return {"nsfw_model": nsfw_model, "nsfw_processor": nsfw_processor}


@torch.inference_mode()
def infer(image_paths, artifacts, sample_prop=1):
    nsfw_model, nsfw_processor = artifacts["nsfw_model"], artifacts["nsfw_processor"]

    frames = [Image.open(image_path) for image_path in image_paths]
    ## sample "sample_prop" proportion of frames uniformly from frames
    frames = frames[::sample_prop]

    scores = [nsfw_label for nsfw_label in run_nsfw(nsfw_model, nsfw_processor, frames)]

    ## average the scores -> count the number of "nsfw" labels
    avg_nsfw_score = sum([1 if score == "nsfw" else 0 for score in scores]) / len(
        scores
    )

    frame_nsfw_results = {
        image_path.split("/")[-1]: score
        for image_path, score in zip(image_paths, scores)
    }

    return {
        "avg_nsfw_score": avg_nsfw_score,
        "frame_nsfw_results": frame_nsfw_results,
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
    parser.add_argument("--nsfw_model_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    image_dir = os.path.join("../../assets/extracted_frames", args.clip_name)
    image_paths = [
        os.path.join(image_dir, image_path)
        for image_path in sorted(os.listdir(image_dir))
    ]
    artifacts = load_artifacts(args)
    results = infer(image_paths=image_paths, artifacts=artifacts)
    print(f"Result for one frame: {results['nsfw_scores'][image_paths[0]]}")
