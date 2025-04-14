from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from typing import Iterator, List, Union
from PIL import Image
from pathlib import Path
import av


def load_nsfw(device):
    MODEL_ID = "Falconsai/nsfw_image_detection"
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    return model, processor


@torch.no_grad()
def run_nsfw(model, processor, image):
    if not isinstance(image, list):
        image = [image]
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    outputs = model(**inputs).logits
    predicted_labels = outputs.argmax(-1)
    return [model.config.id2label[p.cpu().item()] for p in predicted_labels]
