import os
import torch
import argparse
from PIL import Image

from transformers import AutoModelForCausalLM, AutoProcessor
from video_filters.frame_utils import return_frames


def load_artifacts(args):
    repo_id = args.shot_categorizer_repo_id
    model = (
        AutoModelForCausalLM.from_pretrained(
            repo_id, torch_dtype=torch.float16, trust_remote_code=True
        )
        .to("cuda")
        .eval()
    )
    processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)
    return {"model": model, "processor": processor}


@torch.no_grad()
@torch.inference_mode()
def infer(image_paths=None, artifacts=None, frames_list=None):
    model, processor = artifacts["model"], artifacts["processor"]

    prompts = ["<COLOR>", "<LIGHTING>", "<LIGHTING_TYPE>", "<COMPOSITION>"]

    
    if image_paths is None and frames_list is None:
        raise ValueError("Either image_paths or frames_list must be provided.")
    if frames_list is not None:
        frames = [Image.fromarray(frame) for frame in frames_list]
    else:
        frames = [Image.open(image_path) for image_path in image_paths]
    parsed_answers = {frame_idx: [] for frame_idx in (range(len(frames)))}

    for (frame_idx, frame) in enumerate(frames):
        for prompt in prompts:
            inputs = processor(text=prompt, images=frame, return_tensors="pt").to(
                "cuda", torch.float16
            )
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            generated_text = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            parsed_answer = processor.post_process_generation(
                generated_text, task=prompt, image_size=(frame.width, frame.height)
            )
            parsed_answers[frame_idx].append(parsed_answer)

    return {"frame_shot_categorization": parsed_answers}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_name",
        type=str,
        help="Name of clip whose frame(s) to test on.",
        default="trimmed_chameleon",
    )
    parser.add_argument(
        "--shot_categorizer_repo_id",
        type=str,
        default="diffusers/shot-categorizer-v0",
        help="Path to the image.",
    )
    args = parser.parse_args()

    video_path = "/fs/nexus-projects/mt_sec/t2v_curate/assets/split_scenes/(500)_Days_of_Summer_(2_5)_Movie_CLIP_-_Playing_House_(2009)_HD/scene_001.mp4"
    frames = return_frames(clip_path=video_path)

    artifacts = load_artifacts(args)
    results = infer(frames_list=frames, artifacts=artifacts)
    print(f"Result for one frame: {results['frame_shot_categorization'][0]}")
