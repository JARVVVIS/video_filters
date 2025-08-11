import os
import time
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
    return {"aesthetic_model": aesthetic_model, "batch_size": args.batch_size}


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

@torch.no_grad()
@torch.inference_mode()
def infer_batch(frames_list=None, artifacts=None, sample_prop=1):
    if artifacts is None or "aesthetic_model" not in artifacts:
        raise ValueError("artifacts must include 'aesthetic_model'.")
    if frames_list is None or not isinstance(frames_list, list):
        raise ValueError("frames_list must be a list of lists of PIL Images.")
    if sample_prop is None or int(sample_prop) < 1:
        raise ValueError("sample_prop must be an integer >= 1.")

    aesthetic_model = artifacts["aesthetic_model"]
    micro_bs = int(artifacts.get("batch_size", 32))
    sample_prop = int(sample_prop)

    # Subsample each video's frames and track indices for reconstruction
    per_video_sampled = []  # list of lists of frames (sampled) per video
    flat_frames = []  # flattened sampled frames across videos
    flat_map = []  # (video_idx, frame_idx_within_video_after_sampling)

    for vid_idx, vid_frames in enumerate(frames_list):
        if vid_frames is None:
            vid_frames = []
        sampled = vid_frames[::sample_prop]
        per_video_sampled.append(sampled)
        for j, fr in enumerate(sampled):
            flat_frames.append(fr)
            flat_map.append((vid_idx, j))

    # Early out if nothing to score
    if len(flat_frames) == 0:
        return [
            {"avg_aesthetic_score": float("nan"), "frame_aesthetic_scores": {}}
            for _ in frames_list
        ]

    # Run CLIP in micro-batches over ALL sampled frames
    all_scores = []
    for start in range(0, len(flat_frames), micro_bs):
        batch_imgs = flat_frames[start : start + micro_bs]
        batch_scores = run_aesthetic_laion(
            aesthetic_model, batch_imgs
        )  # tensor of shape [B]
        # ensure tensor -> python floats
        all_scores.extend(torch.as_tensor(batch_scores).detach().cpu().tolist())

    # Reassemble per-video scores
    per_video_scores = [[] for _ in frames_list]
    for (vid_idx, local_idx), score in zip(flat_map, all_scores):
        # ensure insertion in order; local_idx already enumerates sampled order
        per_video_scores[vid_idx].append(float(score))

    # Build per-video dicts in original order
    results = []
    for scores in per_video_scores:
        if len(scores) == 0:
            results.append(
                {"avg_aesthetic_score": float("nan"), "frame_aesthetic_scores": {}}
            )
            continue
        frame_aesthetic_scores = {i: s for i, s in enumerate(scores)}
        avg_aesthetic_score = float(sum(scores) / len(scores))
        results.append(
            {
                "avg_aesthetic_score": avg_aesthetic_score,
                "frame_aesthetic_scores": frame_aesthetic_scores,
            }
        )

    return results


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
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    video_path = "/fs/nexus-projects/mt_sec/t2v_curate/assets/split_scenes_long/(500)_Days_of_Summer_(2_5)_Movie_CLIP_-_Playing_House_(2009)_HD/scene_003.mp4"
    assert os.path.exists(video_path), f"Video path {video_path} does not exist."
    frames = return_frames(clip_path=video_path)
    frames_list = [frames] * args.batch_size

    artifacts = load_artifacts(args)

    seq_start_time = time.time()
    for frame_list_idx, frames in enumerate(frames_list):
        results = infer(frames_list=frames, artifacts=artifacts)
        print(
            f"[{frame_list_idx}] Average Aesthetic Score: {results['avg_aesthetic_score']}"
        )
    seq_end_time = time.time()
    print(f"Sequential inference took {seq_end_time - seq_start_time:.2f} seconds.")

    # Run batched inference
    batch_start_time = time.time()
    results = infer_batch(frames_list=frames_list, artifacts=artifacts, sample_prop=1)
    batch_end_time = time.time()
    print(f"Batched inference took {batch_end_time - batch_start_time:.2f} seconds.")
    for frame_list_idx, res in enumerate(results):
        print(
            f"[{frame_list_idx}] Average Aesthetic Score: {res['avg_aesthetic_score']}"
        )