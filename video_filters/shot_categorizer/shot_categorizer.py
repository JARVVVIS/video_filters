import os
import torch
import argparse
from PIL import Image

from transformers import AutoModelForCausalLM, AutoProcessor


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
def infer(image_paths, artifacts):
    model, processor = artifacts["model"], artifacts["processor"]

    prompts = ["<COLOR>", "<LIGHTING>", "<LIGHTING_TYPE>", "<COMPOSITION>"]

    parsed_answers = {image_path.split("/")[-1]: [] for image_path in image_paths}

    for image_path in image_paths:
        image = Image.open(image_path)

        for prompt in prompts:
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(
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
                generated_text, task=prompt, image_size=(image.width, image.height)
            )
            parsed_answers[image_path.split("/")[-1]].append(parsed_answer)

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

    image_dir = os.path.join("../../assets/extracted_frames", args.clip_name)
    image_paths = [
        os.path.join(image_dir, image_path)
        for image_path in sorted(os.listdir(image_dir))
    ]
    artifacts = load_artifacts(args)
    results = infer(image_paths=image_paths, artifacts=artifacts)
    print(f"Result for one frame: {results['shot_categorization'][image_paths[0]]}")
