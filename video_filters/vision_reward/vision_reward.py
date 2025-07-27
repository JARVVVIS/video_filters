## Adapted to directly work on frames from "https://github.com/THUDM/VisionReward/blob/main/inference-video.py"

import io
import json
import numpy as np
import torch
import sys
import types
from torchvision.transforms.functional import rgb_to_grayscale
from video_filters.frame_utils import extract_frames_with_timestamp

# --- NEW imports -----------------------------------------------------------
import os
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------


# Create a module for torchvision.transforms.functional_tensor
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale

# Add this module to sys.modules so other imports can access it
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

from decord import cpu, VideoReader, bridge
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_TYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)


def load_video(source, strategy: str = "chat", num_frames: int = 24):
    """
    Accepts either
      • raw video bytes
      • a filepath to a video ('.mp4', '.mov', …)
      • a directory that holds individual RGB frames
    Returns
      torch.Tensor with shape (C, T, H, W), dtype=torch.uint8
    """
    # -----------------------------------------------------------------------
    # 1) If the caller handed us *bytes* we fall straight back to your
    #    previous decord-based loader.
    # -----------------------------------------------------------------------
    if isinstance(source, (bytes, bytearray)):
        video_bytes = source
        bridge.set_bridge("torch")
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0))
        total = len(vr)

        if strategy == "base":
            frame_ids = np.linspace(0, total - 1, num_frames, dtype=int)
        else:  # "chat"
            # choose at most one frame-id per second
            timestamps = [ts[0] for ts in vr.get_frame_timestamp(np.arange(total))]
            frame_ids = []
            for sec in range(int(max(timestamps)) + 1):
                frame_ids.append(
                    min(range(total), key=lambda i: abs(timestamps[i] - sec))
                )
                if len(frame_ids) >= num_frames:
                    break

        frames = vr.get_batch(frame_ids)  # (T, H, W, C), uint8
        return frames.permute(3, 0, 1, 2)  # (C, T, H, W)

    # -----------------------------------------------------------------------
    # 2) If `source` is a str / Path, decide whether it’s a *video file*
    #    or a *frame directory*.
    # -----------------------------------------------------------------------
    if isinstance(source, (str, Path)):
        source = Path(source)

        # 2-a) FRAME DIRECTORY ------------------------------------------------
        if source.is_dir():
            # Gather image files and keep a deterministic order
            img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            frame_paths = sorted(
                p for p in source.iterdir() if p.suffix.lower() in img_exts
            )
            if not frame_paths:
                raise ValueError(f"No image frames found in {source}")

            total = len(frame_paths)
            # sample the frame indices exactly the same way as above
            if strategy == "base":
                frame_ids = np.linspace(0, total - 1, num_frames, dtype=int)
            else:  # "chat"  (frames are already 1-per-sec, so just stride)
                step = max(total // num_frames, 1)
                frame_ids = list(range(0, total, step))[:num_frames]

            # read & stack selected images
            imgs = [
                torch.as_tensor(
                    np.array(Image.open(frame_paths[i]).convert("RGB")),
                    dtype=torch.uint8,
                )
                for i in frame_ids
            ]  # list of (H, W, C)
            frames = torch.stack(imgs)  # (T, H, W, C)
            return frames.permute(3, 0, 1, 2)  # (C, T, H, W)

        # 2-b) VIDEO FILE ------------------------------------------------------
        elif source.is_file():
            with open(source, "rb") as f:
                return load_video(f.read(), strategy=strategy, num_frames=num_frames)

    # -----------------------------------------------------------------------
    raise TypeError(
        "`source` must be raw bytes, a video filepath, or a directory of frames."
    )


def load_artifacts(args):

    MODEL_PATH = "THUDM/VisionReward-Video"

    if args.filter_prefix is not None:
        QUESTIONS_PATH = f"{args.filter_prefix}/VisionReward_video_qa_select.txt"
        WEIGHT_PATH = f"{args.filter_prefix}/weight.json"
    else:
        QUESTIONS_PATH = "VisionReward_video_qa_select.txt"
        WEIGHT_PATH = "weight.json"

    with open(QUESTIONS_PATH, "r") as f:
        questions = f.readlines()

    with open(WEIGHT_PATH, "r") as f:
        weight = json.load(f)
        weight = np.array(weight)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        # padding_side="left"
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=TORCH_TYPE, trust_remote_code=True
        )
        .eval()
        .to(DEVICE)
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "questions": questions,
        "weight": weight,
        "torch_type": TORCH_TYPE,
    }


def base_inference(
    video_source, query, temperature: float = 0.1, model=None, tokenizer=None
):
    """
    `video_source` may be:
        • video bytes
        • path/to/video.mp4
        • path/to/frame_directory/
    """
    video = load_video(video_source, strategy="chat")
    history = []

    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer,
        query=query,
        images=[video],
        history=history,
        template_version="chat",
    )

    inputs = {
        "input_ids": inputs["input_ids"].unsqueeze(0).to(DEVICE),
        "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(DEVICE),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to(DEVICE),
        "images": [[inputs["images"][0].to(DEVICE).to(TORCH_TYPE)]],
    }

    gen_kwargs = dict(
        max_new_tokens=2048,
        pad_token_id=128002,
        top_k=1,
        top_p=0.1,
        do_sample=False,
        temperature=temperature,
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@torch.inference_mode()
def infer(image_paths=None, artifacts=None, frames_list=None, verbose=False, **kwargs):
    if "video_source" in kwargs:    
        video_source = kwargs['video_source']
    else:
        video_source = image_paths
    
    model, tokenizer, questions, weight = (
        artifacts["model"],
        artifacts["tokenizer"],
        artifacts["questions"],
        artifacts["weight"],
    )

    sub_questions = [question for question in questions if "[[prompt]]" not in question]
    sub_weights = [
        weight[ques_idx]
        for ques_idx, (question) in enumerate(questions)
        if "[[prompt]]" not in question
    ]
    assert len(sub_questions) == len(sub_weights)
    sub_weights = np.array(sub_weights)
    answers = []

    for query in tqdm(sub_questions, "scoring video"):
        answer = base_inference(video_source, query, model=model, tokenizer=tokenizer)
        answers.append(answer)
        if verbose:
            print(f"{query} : {answer}")
    answers = np.array([1 if answer == "yes" else -1 for answer in answers])

    return_dict = {
        "avg_vision_reward": np.mean(answers * sub_weights).item(),
        "frame_wise_rewards": answers,
    }

    return return_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogVLM2-Video CLI Demo")

    parser.add_argument(
        "--quant",
        type=int,
        choices=[4, 8],
        help="Enable 4-bit or 8-bit precision loading",
        default=0,
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to be answered",
        default="Is there a man in the video?",
    )
    parser.add_argument(
        "--score",
        help="Whether to output the score",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--compare",
        help="Whether to compare two videos",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    args.filter_prefix = None

    video1 = "test1.mp4"
    assert os.path.exists(video1), f"Video {video1} not found"

    model_dict = load_artifacts(args)
    
    score_1 = infer(video1, model_dict)
    print(f"Score via video: {score_1}")

    score_1 = infer(artifacts=model_dict, video_source=video1)
    print(f"Score via kwargs: {score_1}")