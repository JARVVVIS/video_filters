import cv2
import os
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def trim_video_end(video_path: str, discard_last: int, output_path: str = None) -> bool:
    """
    Trim the last X seconds from a video and optionally overwrite the original.
    
    Args:
        video_path: Path to the input video file
        discard_last: Number of seconds to discard from the end
        output_path: Path for output video (if None, overwrites input)
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    if discard_last <= 0:
        logger.info(f"No trimming needed (discard_last={discard_last})")
        return True
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate video duration and new duration
    total_duration = total_frames / fps
    new_duration = total_duration - discard_last
    
    if new_duration <= 0:
        logger.error(f"Cannot discard {discard_last} seconds from {total_duration:.1f}s video")
        cap.release()
        return False
    
    # Calculate how many frames to keep
    frames_to_keep = int(new_duration * fps)
    
    logger.info(f"Trimming {video_path}:")
    logger.info(f"  Original: {total_duration:.1f}s ({total_frames} frames)")
    logger.info(f"  New: {new_duration:.1f}s ({frames_to_keep} frames)")
    logger.info(f"  Discarding: {discard_last}s ({total_frames - frames_to_keep} frames)")
    
    # Create temporary output file if needed
    temp_output = None
    if output_path is None:
        temp_fd, temp_output = tempfile.mkstemp(suffix='.mp4', dir=os.path.dirname(video_path))
        os.close(temp_fd)
        output_path = temp_output
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        logger.error("Cannot create output video writer")
        cap.release()
        return False
    
    # Process frames (only up to frames_to_keep)
    frame_count = 0
    
    while frame_count < frames_to_keep:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Reached end of video at frame {frame_count} (expected {frames_to_keep})")
            break
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{frames_to_keep} frames")
    
    # Clean up
    cap.release()
    out.release()
    
    # Replace original file if needed
    if temp_output:
        os.replace(temp_output, video_path)
        logger.info(f"Overwrote original video: {video_path}")
    else:
        logger.info(f"Created trimmed video: {output_path}")
    
    return True


def batch_trim_videos(video_folder_path: str, discard_last: int, file_pattern: str = "*.mp4") -> None:
    """
    Batch trim the last X seconds from all videos in a folder.
    
    Args:
        video_folder_path: Path to folder containing videos
        discard_last: Number of seconds to discard from each video
        file_pattern: File pattern to match (default: "*.mp4")
    """
    import glob
    
    video_files = glob.glob(os.path.join(video_folder_path, file_pattern))
    print(f"Found {len(video_files)} video files to trim")
    
    successful = 0
    failed = 0
    
    for video_file in video_files:
        print(f"Trimming: {os.path.basename(video_file)}")
        try:
            success = trim_video_end(video_file, discard_last)
            if success:
                successful += 1
                print(f"✓ Successfully trimmed: {os.path.basename(video_file)}")
            else:
                failed += 1
                print(f"✗ Failed to trim: {os.path.basename(video_file)}")
        except Exception as e:
            failed += 1
            print(f"✗ Error trimming {os.path.basename(video_file)}: {str(e)}")
    
    print(f"\nBatch trimming complete:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(video_files)}")


def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video"}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "duration_seconds": duration,
        "total_frames": total_frames,
        "fps": fps,
        "width": frame_width,
        "height": frame_height,
        "resolution": f"{frame_width}x{frame_height}"
    }


# Example usage
if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"
    
    # Get video info first
    info = get_video_info(video_path)
    print(f"Video info: {info}")
    
    # Trim last 5 seconds
    success = trim_video_end(video_path, discard_last=5)
    print(f"Trimming {'successful' if success else 'failed'}")
    
    # Batch trim all videos in a folder
    # batch_trim_videos("/path/to/video/folder", discard_last=3)