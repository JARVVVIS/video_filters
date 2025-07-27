import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from typing import Iterator, List, Union
from PIL import Image
from pathlib import Path
import av


def load_watermark_laion(device, model_path):
    transforms = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )
    if model_path is None:
        model_path = hf_hub_download(
            "finetrainers/laion-watermark-detection", "watermark_model_v1.pt"
        )
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    return model, transforms


@torch.no_grad
def run_watermark_laion(model, transforms, image):
    if not isinstance(image, list):
        image = [image]

    pixel_values = torch.stack([transforms(_image) for _image in image]).to("cuda")
    return nn.functional.softmax(model(pixel_values), dim=1)[:, 0]
