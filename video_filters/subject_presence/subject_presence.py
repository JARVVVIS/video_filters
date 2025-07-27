# requirements (conda or pip):
# pip install ultralytics opencv-python pillow torch torchvision segment_anything
import argparse, os

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import numpy as np, torch, cv2, json
import cv2, random, pathlib
from PIL import Image
from video_filters.frame_utils import return_frames

PALETTE = {
    "humans":  (255,  0,  0),
    "animals": (  0,255,  0),
    "vehicles":(  0,  0,255),
    "food":    (255,255,  0),
    "others":  (255,165,  0),
}

def annotate_and_save(img_bgr, det_out, save_path, max_boxes=100, CLASS_NAMES=None):
    for xyxy, cls, conf in zip(det_out.boxes.xyxy.cpu().numpy(),
                                det_out.boxes.cls.cpu().numpy().astype(int),
                                det_out.boxes.conf.cpu().numpy()):
        cat = bucket(cls, CLASS_NAMES)
        c = PALETTE.get(cat, (255,255,255))
        x1,y1,x2,y2 = map(int, xyxy)
        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), c, 2)
        cv2.putText(img_bgr,
                    f"{cat}:{conf:.2f}",
                    (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, c, 1,
                    lineType=cv2.LINE_AA)
    cv2.imwrite(str(save_path), img_bgr)

# ---------- loading ----------
def load_artifacts(args):
    # SAM
    ckpt = "video_filters/sam_weights/sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    # YOLO-v8n for speed; switch to v8m/v8x for higher accuracy
    yolo = YOLO("yolov8n.pt")          # COCO-trained weights
    yolo.to("cuda" if torch.cuda.is_available() else "cpu")
    return {"predictor": predictor, "detector": yolo}

# ---------- categories ----------
CATEGORY_KEYWORDS = {
    "humans"  : {"person", "man", "woman", "boy", "girl"},
    "animals" : {"cat", "dog", "bird", "horse", "sheep", "cow",
                 "elephant", "bear", "zebra", "giraffe", "duck",
                 "animal", "monkey", "panda", "tiger", "lion"},
    "vehicles": {"car", "truck", "bus", "train", "motorcycle", "bicycle",
                 "plane", "boat", "ship", "van"},
    "food"    : {"banana", "apple", "sandwich", "orange", "broccoli",
                 "carrot", "pizza", "cake", "hot dog", "donut"},
    # add more buckets / keywords as you wish
}

def bucket(cls_id: int, CLASS_NAMES) -> str:
    label = CLASS_NAMES[int(cls_id)].lower()
    for cat, kw_set in CATEGORY_KEYWORDS.items():
        if any(k in label for k in kw_set):
            return cat
    return "others"

# ---------- inference ----------
SAVE_DIR = pathlib.Path("debug_frames")
SAVE_DIR.mkdir(exist_ok=True)

@torch.inference_mode()
def infer(image_paths=None, artifacts=None, frames_list=None, conf_thres=0.25):
    pred = artifacts["predictor"]
    det = artifacts["detector"]
    CLASS_NAMES = det.names
    per_frame = {}
    
    if image_paths is None and frames_list is None:
        raise ValueError("Either image_paths or frames_list must be provided.")
    if frames_list is not None:
        frames = [Image.fromarray(frame) for frame in frames_list]
    else:
        frames = [Image.open(image_path) for image_path in image_paths]
        
    for frame_idx, frame in enumerate(frames):
        w, h = frame.size  # PIL Image.size returns (width, height)
        # 1️⃣  run detector
        det_out = det(frame, conf=conf_thres, verbose=False)[0]
        
        if frame_idx < 5:  # save first 5 frames (tweak as you like)
            # Convert PIL Image to numpy array for OpenCV operations
            frame_array = np.array(frame)
            annotate_and_save(frame_array.copy(), det_out,
                          SAVE_DIR / f"{frame_idx}_annot.png", CLASS_NAMES=CLASS_NAMES)
        
        # 2️⃣  Count both raw labels and bucketed categories
        cnts = {}  # raw class label counts
        bucketed_cnts = {"humans": 0, "animals": 0, "vehicles": 0, "food": 0, "others": 0}
        
        for xyxy, cls in zip(det_out.boxes.xyxy.cpu().numpy(),
                              det_out.boxes.cls.cpu().numpy().astype(int)):
            # Count by raw label
            label = CLASS_NAMES[int(cls)].lower()
            if label not in cnts:
                cnts[label] = 0
            cnts[label] += 1
            
            # Count by bucketed category
            category = bucket(cls, CLASS_NAMES)
            bucketed_cnts[category] += 1
            
        cnts["total"] = sum(cnts.values())
        bucketed_cnts["total"] = sum(bucketed_cnts.values())
        
        per_frame[frame_idx] = {
            "raw_counts": cnts,
            "bucketed_counts": bucketed_cnts
        }
    
    # dataset-level stats - use bucketed counts for averages
    avg_total = np.mean([v["bucketed_counts"]["total"] for v in per_frame.values()])
    avg_humans = np.mean([v["bucketed_counts"]["humans"] for v in per_frame.values()])
    avg_animals = np.mean([v["bucketed_counts"]["animals"] for v in per_frame.values()])
    
    return {
        "avg_subjects": avg_total,
        "avg_humans": avg_humans,
        "avg_animals": avg_animals,
        "frame_counts": per_frame,
    }


# ---------- driver ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", default="../../assets/extracted_frames_cinepile")
    ap.add_argument("--clip_name", default=None)
    args = ap.parse_args()
    
    
    video_path = "/fs/nexus-projects/mt_sec/t2v_curate/assets/split_scenes/(500)_Days_of_Summer_(2_5)_Movie_CLIP_-_Playing_House_(2009)_HD/scene_001.mp4"
    frames = return_frames(clip_path=video_path)
    
    
    
    art = load_artifacts(args)
    res = infer(frames_list=frames, artifacts=art)
    print(json.dumps(res, indent=2))