# requirements (conda or pip):
# pip install ultralytics opencv-python pillow torch torchvision segment_anything
import argparse, os, json, pathlib, time
from time import perf_counter
import urllib
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import numpy as np, torch, cv2, random
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
    if not os.path.exists(ckpt):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        print(f"Downloading SAM checkpoint to {ckpt}...")
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        try:
            # Download the file
            urllib.request.urlretrieve(url, ckpt)
            print(f"Successfully downloaded SAM checkpoint to {ckpt}")
        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
            return False

    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    # YOLO-v8n for speed; switch to v8m/v8x for higher accuracy
    yolo = YOLO("yolov8n.pt")          # COCO-trained weights
    yolo.to("cuda" if torch.cuda.is_available() else "cpu")
    return {"predictor": predictor, "detector": yolo, "batch_size": args.batch_size}

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


SAVE_DIR = pathlib.Path("debug_frames")
SAVE_DIR.mkdir(exist_ok=True)


@torch.inference_mode()
def infer(
    image_paths=None, artifacts=None, frames_list=None, conf_thres=0.25, save_annot_n=5
):
    det = artifacts["detector"]
    CLASS_NAMES = det.names
    per_frame = {}

    if image_paths is None and frames_list is None:
        raise ValueError("Either image_paths or frames_list must be provided.")
    if frames_list is not None:
        frames = [
            Image.fromarray(frame) if not isinstance(frame, Image.Image) else frame
            for frame in frames_list
        ]
    else:
        frames = [Image.open(image_path) for image_path in image_paths]

    for frame_idx, frame in enumerate(frames):
        det_out = det(frame, conf=conf_thres, verbose=False)[0]

        if frame_idx < save_annot_n:
            frame_array = np.array(frame)  # RGB; fine for visualization
            annotate_and_save(
                frame_array.copy(),
                det_out,
                SAVE_DIR / f"{frame_idx}_annot.png",
                CLASS_NAMES=CLASS_NAMES,
            )

        cnts = {}
        bucketed_cnts = {
            "humans": 0,
            "animals": 0,
            "vehicles": 0,
            "food": 0,
            "others": 0,
        }

        for xyxy, cls in zip(
            det_out.boxes.xyxy.cpu().numpy(),
            det_out.boxes.cls.cpu().numpy().astype(int),
        ):
            label = CLASS_NAMES[int(cls)].lower()
            cnts[label] = cnts.get(label, 0) + 1
            category = bucket(cls, CLASS_NAMES)
            bucketed_cnts[category] += 1

        cnts["total"] = sum(v for k, v in cnts.items() if k != "total")
        bucketed_cnts["total"] = sum(bucketed_cnts.values())

        per_frame[frame_idx] = {
            "raw_counts": cnts,
            "bucketed_counts": bucketed_cnts
        }

    avg_total = (
        float(np.mean([v["bucketed_counts"]["total"] for v in per_frame.values()]))
        if per_frame
        else 0.0
    )
    avg_humans = (
        float(np.mean([v["bucketed_counts"]["humans"] for v in per_frame.values()]))
        if per_frame
        else 0.0
    )
    avg_animals = (
        float(np.mean([v["bucketed_counts"]["animals"] for v in per_frame.values()]))
        if per_frame
        else 0.0
    )

    return {
        "avg_subjects": avg_total,
        "avg_humans": avg_humans,
        "avg_animals": avg_animals,
        "frame_counts": per_frame,
    }


# ---------- inference (multi-video; micro-batched per video) ----------
@torch.inference_mode()
def infer_batch(frames_list=None, artifacts=None, conf_thres=0.25, save_annot_n=5):
    if frames_list is None or not isinstance(frames_list, list):
        raise ValueError("frames_list must be a list of lists of frames.")

    det = artifacts["detector"]
    CLASS_NAMES = det.names
    micro_bs = max(1, int(artifacts.get("batch_size", 16)))

    results_per_video = []

    for vid_idx, vid_frames in enumerate(frames_list):
        # Normalize to PIL
        frames = [
            Image.fromarray(f) if not isinstance(f, Image.Image) else f
            for f in (vid_frames or [])
        ]

        per_frame = {}
        # Micro-batch frames for this ONE video
        for start in range(0, len(frames), micro_bs):
            batch_frames = frames[start : start + micro_bs]
            # YOLO supports list input -> batched inference
            det_outs = det(batch_frames, conf=conf_thres, verbose=False)
            # Ensure we have a list (ultralytics may yield a generator)
            det_outs = list(det_outs)

            for i, (frame, det_out) in enumerate(zip(batch_frames, det_outs)):
                frame_idx = start + i

                if frame_idx < save_annot_n:
                    arr = np.array(frame)
                    annotate_and_save(
                        arr.copy(),
                        det_out,
                        SAVE_DIR / f"v{vid_idx}_f{frame_idx}_annot.png",
                        CLASS_NAMES=CLASS_NAMES,
                    )

                cnts = {}
                bucketed_cnts = {
                    "humans": 0,
                    "animals": 0,
                    "vehicles": 0,
                    "food": 0,
                    "others": 0,
                }

                if det_out.boxes is not None and det_out.boxes.xyxy is not None:
                    for xyxy, cls in zip(
                        det_out.boxes.xyxy.cpu().numpy(),
                        det_out.boxes.cls.cpu().numpy().astype(int),
                    ):
                        label = CLASS_NAMES[int(cls)].lower()
                        cnts[label] = cnts.get(label, 0) + 1
                        category = bucket(cls, CLASS_NAMES)
                        bucketed_cnts[category] += 1

                cnts["total"] = sum(v for k, v in cnts.items() if k != "total")
                bucketed_cnts["total"] = sum(bucketed_cnts.values())

                per_frame[frame_idx] = {
                    "raw_counts": cnts,
                    "bucketed_counts": bucketed_cnts,
                }

        # Aggregate for this video
        if per_frame:
            avg_total = float(
                np.mean([v["bucketed_counts"]["total"] for v in per_frame.values()])
            )
            avg_humans = float(
                np.mean([v["bucketed_counts"]["humans"] for v in per_frame.values()])
            )
            avg_animals = float(
                np.mean([v["bucketed_counts"]["animals"] for v in per_frame.values()])
            )
        else:
            avg_total = avg_humans = avg_animals = 0.0

        results_per_video.append(
            {
                "avg_subjects": avg_total,
                "avg_humans": avg_humans,
                "avg_animals": avg_animals,
                "frame_counts": per_frame,
            }
        )

    return results_per_video


# ---------- driver ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", default="../../assets/extracted_frames_cinepile")
    ap.add_argument("--clip_name", default=None)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Micro-batch size (frames per forward) within a single video",
    )
    ap.add_argument(
        "--video_batch",
        type=int,
        default=4,
        help="How many copies of the same video to simulate for batched testing",
    )
    ap.add_argument("--conf_thres", type=float, default=0.25)
    ap.add_argument(
        "--save_annot_n",
        type=int,
        default=5,
        help="How many frames (per video) to annotate & save",
    )
    ap.add_argument(
        "--with_sam",
        action="store_true",
        help="Load SAM predictor (unused in this script)",
    )
    args = ap.parse_args()

    # Example source: one real video (repeat it to simulate multiple videos)
    video_path = "/fs/nexus-projects/mt_sec/t2v_curate/assets/split_scenes/short_(500)_Days_of_Summer_(2_5)_Movie_CLIP_-_Playing_House_(2009)_HD/short_scene_001.mp4"
    assert os.path.exists(video_path), f"Video path {video_path} does not exist."
    frames = return_frames(clip_path=video_path)  # List[np.ndarray]
    frames_list = [frames] * max(1, args.video_batch)  # List[List[np.ndarray]]

    art = load_artifacts(args)

    # --- Baseline (single-video, unbatched over frames) ---
    t0 = perf_counter()
    baseline = infer(
        frames_list=frames,
        artifacts=art,
        conf_thres=args.conf_thres,
        save_annot_n=args.save_annot_n,
    )
    t1 = perf_counter()
    baseline_time = t1 - t0
    baseline_fps = (len(frames) / baseline_time) if baseline_time > 0 else float("inf")

    # --- Batched (multi-video list; micro-batched per video) ---
    t2 = perf_counter()
    batched = infer_batch(
        frames_list=frames_list,
        artifacts=art,
        conf_thres=args.conf_thres,
        save_annot_n=args.save_annot_n,
    )
    t3 = perf_counter()
    batched_time = t3 - t2
    total_frames = len(frames) * len(frames_list)
    batched_fps = (total_frames / batched_time) if batched_time > 0 else float("inf")

    # --- Report ---
    print("\n===== Benchmark =====")
    print(f"Frames per video         : {len(frames)}")
    print(f"Number of videos (simul.): {len(frames_list)}")
    print(f"Micro-batch size (frames): {args.batch_size}")
    print(f"YOLO conf_thres          : {args.conf_thres}")
    print(f"Baseline time (s)        : {baseline_time:.3f}")
    print(f"Baseline FPS (1 video)   : {baseline_fps:.2f}")
    print(f"Batched time (s)         : {batched_time:.3f}")
    print(f"Batched FPS (all videos) : {batched_fps:.2f}")
    if baseline_fps > 0:
        print(
            f"Speedup (vs per-video baseline, normalized per frame): {batched_fps / baseline_fps:.2f}x"
        )
    print("===== End Benchmark =====\n")

    # Print a compact JSON sample so stdout isn't huge
    print("Sample (video 0) result:")
    print(json.dumps(batched[0], indent=2)[:2000])
