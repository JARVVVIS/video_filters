from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import torch


def load_artifacts(args):
    checkpoint = "/BRAIN/adv-robustness/work/t2v_curate/video_filters/sam_weights/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    # Initialize the SAM model and predictor.
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    predictor = SamPredictor(sam)

    return {"sam": sam, "predictor": predictor}


@torch.inference_mode()
def infer(image_paths, artifacts, sample_prop=1):
    sam, predictor = artifacts["sam"], artifacts["predictor"]

    frame_num_subjects = {image_path.split("/")[-1]: None for image_path in image_paths}

    # Load the image using PIL.
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        # Open the image and convert it to a NumPy array.
        image_np = np.array(image)  # assume image_np is (H, W, 3) in RGB

        predictor.set_image(image_np)

        # Create a bounding box that covers the entire image.
        width, height = image.size
        box = np.array([0, 0, width, height]).reshape(1, 4)

        # Use the bounding box prompt to get segmentation masks.
        # Set multimask_output=True to get multiple mask proposals.
        masks, scores, logits = predictor.predict(box=box, multimask_output=True)

        frame_num_subjects[image_path.split("/")[-1]] = len(masks)

    ## compute average number of subjects
    avg_num_subjects = sum(frame_num_subjects.values()) / len(frame_num_subjects)

    return {
        "avg_num_subjects": avg_num_subjects,
        "frame_num_subjects": frame_num_subjects,
    }


def main():
    pass


if __name__ == "__main__":
    main()
