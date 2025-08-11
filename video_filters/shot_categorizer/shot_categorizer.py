import os
import torch
import argparse
from PIL import Image
import time
from transformers import AutoModelForCausalLM, AutoProcessor
from video_filters.frame_utils import return_frames
import torch.nn.functional as F


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
    return {"model": model, "processor": processor, "batch_size": args.batch_size}


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


def _ensure_pad_token(processor):
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token


def _collate_processor_outputs(per_item_inputs, processor, device="cuda"):
    out = {}

    # input_ids: [1, L] -> pad to max L -> [B, L] (keep dtype long)
    if "input_ids" in per_item_inputs[0]:
        pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", 0)
        max_len = max(x["input_ids"].shape[-1] for x in per_item_inputs)
        ids = []
        for x in per_item_inputs:
            t = x["input_ids"]  # [1, L]
            L = t.shape[-1]
            if L < max_len:
                t = F.pad(t, (0, max_len - L), value=pad_id)
            ids.append(t)
        out["input_ids"] = torch.cat(ids, dim=0).to(device)

    # pixel_values: [1, C, H, W] -> [B, C, H, W] (half)
    if "pixel_values" in per_item_inputs[0]:
        pv = torch.cat([x["pixel_values"] for x in per_item_inputs], dim=0)
        out["pixel_values"] = pv.to(device=device, dtype=torch.float16)

    return out


@torch.no_grad()
@torch.inference_mode()
def infer_batch(frames_list=None, artifacts=None, sample_prop=1):
    if artifacts is None or "model" not in artifacts or "processor" not in artifacts:
        raise ValueError("artifacts must include 'model' and 'processor'.")
    if frames_list is None or not isinstance(frames_list, list):
        raise ValueError("frames_list must be a list of lists of frames.")
    if int(sample_prop) < 1:
        raise ValueError("sample_prop must be an integer >= 1.")

    model, processor = artifacts["model"], artifacts["processor"]
    _ensure_pad_token(processor)
    micro_bs = int(artifacts.get("micro_batch", artifacts.get("batch_size", 16)))
    sample_prop = int(sample_prop)

    prompts = ["<COLOR>", "<LIGHTING>", "<LIGHTING_TYPE>", "<COMPOSITION>"]
    num_tasks_per_frame = len(prompts)

    def _to_pil(f):
        return f if isinstance(f, Image.Image) else Image.fromarray(f)

    outputs = []

    # one video at a time
    for vid_frames in frames_list:
        pil_frames = [_to_pil(f) for f in (vid_frames or [])]
        sampled = pil_frames[::sample_prop]

        per_frame_answers = {
            j: [None] * num_tasks_per_frame for j in range(len(sampled))
        }

        # build tasks: (frame_idx, prompt, frame, (W,H))
        tasks = []
        for j, frame in enumerate(sampled):
            wh = (frame.width, frame.height)
            for p in prompts:
                tasks.append((j, p, frame, wh))

        # micro-batch inside this video
        for start in range(0, len(tasks), micro_bs):
            chunk = tasks[start : start + micro_bs]
            batch_texts = [t[1] for t in chunk]
            batch_images = [t[2] for t in chunk]
            sizes = [t[3] for t in chunk]

            # per-sample -> collate (no attention_mask forwarded)
            per_item = [
                processor(text=txt, images=img, return_tensors="pt")
                for txt, img in zip(batch_texts, batch_images)
            ]
            inputs = _collate_processor_outputs(per_item, processor, device="cuda")

            # DO NOT pass attention_mask; Florence-2 handles internal masks
            generated_ids = model.generate(
                input_ids=inputs.get("input_ids"),
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            generated_texts = processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )

            # place results
            for (local_fidx, prompt_str, _, (w, h)), gen_txt in zip(
                chunk, generated_texts
            ):
                parsed = processor.post_process_generation(
                    gen_txt, task=prompt_str, image_size=(w, h)
                )
                ## remove the <pad> token in "prased's key" if it exists; there can be multiple <pad> tokens existing
                for k, v in parsed.items():
                    if isinstance(v, str) and "<pad>" in v:
                        parsed[k] = v.replace("<pad>", "").strip()
                p_idx = prompts.index(prompt_str)
                per_frame_answers[local_fidx][p_idx] = parsed

        outputs.append({"frame_shot_categorization": per_frame_answers})
    return outputs


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
    parser.add_argument("--batch_size", type=int, default=8)
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
            f"[{frame_list_idx}] Average Shot-Cat: {results['frame_shot_categorization'][0]}"
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
            f"[{frame_list_idx}] Average Shot-Cat: {res['frame_shot_categorization'][0]}"
        )