import av
import os
import cv2
import math
import argparse
from PIL import Image
from pathlib import Path
from typing import Iterator, List, Union


def resize_image(img, target_size):
    # Open the image
    # Get original dimensions
    original_width, original_height = img.size

    # Calculate the ratio of the target size to the original size
    target_width, target_height = target_size
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    # Use the smaller ratio to ensure the image fits within target dimensions
    ratio = min(width_ratio, height_ratio)

    # Calculate new dimensions
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    print(
        f"Image resized from {original_width}x{original_height} to {new_width}x{new_height}"
    )
    return resized_img


def return_frames(clip_path, desired_fps=None, discard_last=None):
    frames_list = []
    video = cv2.VideoCapture(clip_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video at path: {clip_path}")
        return frames_list  # Return empty list instead of None
    
    try:
        real_fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / real_fps if real_fps > 0 else 0
        
        if desired_fps and desired_fps > 0:
            time_step = 1.0 / desired_fps
        else:
            time_step = 1.0
        
        current_time = 0.0
        discard_message_shown = False
        
        while current_time < duration:  # Changed <= to 
            frame_number = int(round(current_time * real_fps))
            
            if (discard_last is not None) and (current_time > duration - discard_last):
                if not discard_message_shown:
                    print("=" * 50)
                    print(f"Discarding last {discard_last} seconds of the video.")
                    print("=" * 50)
                    discard_message_shown = True
                break
            
            if frame_number >= total_frames:
                break
            
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = video.read()
            
            if not success:
                print(f"Warning: Could not read frame at time {current_time:.2f} seconds.")
            else:
                frames_list.append(frame)
            
            current_time += time_step  # CRITICAL: This was missing!
    
    finally:
        video.release()  # Always release the resource
    
    return frames_list

def extract_frames_with_timestamp(
    clip_path, frames_save_dir, desired_fps=None, discard_last=None
):
    FRAME_PREFIX = "frame_"
    frames_list = []

    # Ensure the save directory exists
    if not os.path.exists(frames_save_dir):
        os.makedirs(frames_save_dir)

    # Open the video file
    video = cv2.VideoCapture(clip_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Get actual video FPS and total frames
    real_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # Compute total duration in seconds (float)
    duration = total_frames / real_fps if real_fps > 0 else 0

    # Decide on time-step based on desired_fps or default to 1-second intervals
    if desired_fps and desired_fps > 0:
        time_step = 1.0 / desired_fps
    else:
        time_step = 1.0  # default: extract one frame per second

    current_time = 0.0
    while current_time <= duration:
        frame_number = int(round(current_time * real_fps))

        ## discard frames from the last 25 seconds of the video
        if (discard_last is not None) and (current_time > duration - discard_last):
            print("=" * 50)
            print(f"Discarding last {discard_last} seconds of the video.")
            print("=" * 50)
            break

        if frame_number >= total_frames:
            break

        # Set the video position to the desired frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video.read()

        if not success:
            print(f"Warning: Could not read frame at time {current_time:.2f} seconds.")
            current_time += time_step
            continue

        # Convert current_time (in seconds) to MM:SS for the filename
        total_seconds = int(math.floor(current_time))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        timestamp = f"{minutes:02d}:{seconds:02d}"

        # Save the frame with timestamp in the filename
        filename = os.path.join(frames_save_dir, f"{FRAME_PREFIX}{timestamp}.png")
        cv2.imwrite(filename, frame)

        # Move to the next time-step
        current_time += time_step
        frames_list.append(frame)

    # Release the video capture object
    video.release()
    print(f"Extracted frames to '{frames_save_dir}'.")


def get_key_frames(path: Union[Path, str]) -> List[Image.Image]:
    frames = []
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONKEY"
    for _, frame in enumerate(container.decode(stream)):
        frames.append(frame.to_image())
    container.close()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_name",
        type=str,
        default="trimmed_chameleon",
    )
    parser.add_argument("--desired_fps", type=int, default=None)
    parser.add_argument(
        "--frames_save_dir", type=str, default="assets/extracted_frames"
    )
    args = parser.parse_args()

    clip_name = args.clip_name
    clip_path = os.path.join("../assets/clips", clip_name + ".mp4")
    desired_fps = args.desired_fps
    frames_save_dir = os.path.join(args.frames_save_dir, clip_name)

    extract_frames_with_timestamp(clip_path, frames_save_dir, desired_fps)


if __name__ == "__main__":
    main()
