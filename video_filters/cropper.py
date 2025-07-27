import cv2
import numpy as np
import os
import tempfile
from typing import Tuple, Optional
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleWatermarkCropper:
    """
    Simple watermark cropper that detects and removes black regions at top/bottom of videos.
    Much more reliable than complex detection algorithms.
    """
    
    def __init__(self, 
                 darkness_threshold: int = 50,
                 black_pixel_ratio: float = 0.7,
                 non_black_pixel_ratio: float = 0.80):
        """
        Initialize the simple watermark cropper.
        
        Args:
            darkness_threshold: Pixel intensity threshold for "black" pixels
            black_pixel_ratio: Ratio of pixels that must be black to consider region as watermark
            non_black_pixel_ratio: Ratio of pixels that must be non-black to stop cropping
        """
        self.darkness_threshold = darkness_threshold
        self.black_pixel_ratio = black_pixel_ratio
        self.non_black_pixel_ratio = non_black_pixel_ratio
    
    def check_black_region(self, frame: np.ndarray, position: str = 'bottom', 
                          percent: float = 10.0) -> bool:
        """
        Check if a specific region of the frame is predominantly black.
        
        Args:
            frame: Input frame as numpy array
            position: 'top' or 'bottom' 
            percent: Percentage of frame height to check
            
        Returns:
            True if region is predominantly black, False otherwise
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        region_height = int(height * percent / 100)
        
        if position == 'bottom':
            region = gray[height - region_height:, :]
        else:  # top
            region = gray[:region_height, :]
        
        # Count black pixels
        black_pixels = np.sum(region < self.darkness_threshold)
        total_pixels = region.size
        
        black_ratio = black_pixels / total_pixels
        
        logger.debug(f"Region {position} {percent}%: {black_ratio:.3f} black ratio")
        
        return black_ratio >= self.black_pixel_ratio
    
    def find_crop_boundary(self, frame: np.ndarray, position: str = 'bottom', 
                          start_percent: float = 1.0, max_percent: float = 30.0, 
                          step_percent: float = 0.5) -> Optional[int]:
        """
        Find the optimal crop boundary by incrementally checking for non-black rows.
        
        Args:
            frame: Input frame as numpy array
            position: 'top' or 'bottom'
            start_percent: Starting percentage to check from
            max_percent: Maximum percentage to check (safety limit)
            step_percent: Step size for incremental checking
            
        Returns:
            Pixel position where to crop, or None if no boundary found
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        current_percent = start_percent
        
        while current_percent <= max_percent:
            region_height = int(height * current_percent / 100)
            
            if position == 'bottom':
                # Check from bottom up
                check_row_idx = height - region_height
                if check_row_idx < 0:
                    break
                row = gray[check_row_idx, :]
            else:  # top
                # Check from top down
                check_row_idx = region_height - 1
                if check_row_idx >= height:
                    break
                row = gray[check_row_idx, :]
            
            # Count non-black pixels in this row
            non_black_pixels = np.sum(row >= self.darkness_threshold)
            non_black_ratio = non_black_pixels / width
            
            logger.debug(f"[Position: {position}] Row at {current_percent}% ({check_row_idx}): {non_black_ratio:.3f} non-black ratio")
                        
            # If we find a row with mostly non-black pixels, this is our boundary
            if (non_black_ratio >= self.non_black_pixel_ratio):
                logger.info(f"Found crop boundary at {current_percent}% ({check_row_idx} pixels)")
                return check_row_idx
            
            current_percent += step_percent
        
        logger.info(f"No clear boundary found within {max_percent}% limit")
        # return the predifined boundary if no clear boundary found
        if position == 'bottom':
            ## discard 15% of the bottom
            return height - int(height * 0.15)
        else:
            ## discard 10% of the top
            return int(height * 0.10)
        
    
    def analyze_frame_for_cropping(self, frame: np.ndarray, 
                                 check_top: bool = True, check_bottom: bool = True,
                                 simple_percent: Optional[float] = None) -> Tuple[Optional[int], Optional[int]]:
        """
        Analyze a frame to determine crop boundaries.
        
        Args:
            frame: Input frame
            check_top: Whether to check for top watermarks
            check_bottom: Whether to check for bottom watermarks
            simple_percent: If provided, use simple percentage check instead of smart boundary finding
            
        Returns:
            Tuple of (top_crop_line, bottom_crop_line) in pixels. None means no crop needed.
        """
        height = frame.shape[0]
        top_crop = None
        bottom_crop = None
        
        if simple_percent:
            # Simple percentage-based cropping
            if check_bottom and self.check_black_region(frame, 'bottom', simple_percent):
                bottom_crop = height - int(height * simple_percent / 100)
                logger.info(f"Bottom {simple_percent}% is black, cropping to {bottom_crop}")
            
            if check_top and self.check_black_region(frame, 'top', simple_percent):
                top_crop = int(height * simple_percent / 100)
                logger.info(f"Top {simple_percent}% is black, cropping from {top_crop}")
        else:
            # Smart boundary finding
            if check_bottom:
                bottom_boundary = self.find_crop_boundary(frame, 'bottom')
                if bottom_boundary is not None:
                    bottom_crop = bottom_boundary
            
            if check_top:
                top_boundary = self.find_crop_boundary(frame, 'top')
                if top_boundary is not None:
                    top_crop = top_boundary
        
        return top_crop, bottom_crop
    
    def crop_video(self, input_path: str, output_path: str = None,
                   check_top: bool = True, check_bottom: bool = True,
                   simple_percent: Optional[float] = None,
                   sample_frames: int = 5) -> bool:
        """
        Crop video to remove black watermark regions.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video (if None, overwrites input)
            check_top: Whether to check for top watermarks
            check_bottom: Whether to check for bottom watermarks
            simple_percent: If provided, use simple percentage check (e.g., 10.0 for 10%)
            sample_frames: Number of frames to analyze for consistency
            
        Returns:
            True if successful, False otherwise
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames to determine crop boundaries
        frame_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
        
        top_crops = []
        bottom_crops = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            top_crop, bottom_crop = self.analyze_frame_for_cropping(
                frame, check_top, check_bottom, simple_percent
            )
            
            if top_crop is not None:
                top_crops.append(top_crop)
            if bottom_crop is not None:
                bottom_crops.append(bottom_crop)
        
        # Determine final crop boundaries (use median for robustness)
        final_top_crop = None
        final_bottom_crop = None
        
        if top_crops:
            final_top_crop = int(np.median(top_crops))
            logger.info(f"Final top crop: {final_top_crop} pixels")
        
        if bottom_crops:
            final_bottom_crop = int(np.median(bottom_crops))
            logger.info(f"Final bottom crop: {final_bottom_crop} pixels")
        
        # If no cropping needed, return early
        if final_top_crop is None and final_bottom_crop is None:
            logger.info(f"No black regions detected in {input_path}, skipping crop")
            cap.release()
            return True
        
        # Calculate final crop region
        start_y = final_top_crop if final_top_crop is not None else 0
        end_y = final_bottom_crop if final_bottom_crop is not None else frame_height
        
        new_height = end_y - start_y
        new_width = frame_width
        
        if new_height <= 0:
            logger.error("Invalid crop region calculated")
            cap.release()
            return False
        
        logger.info(f"Cropping {input_path}: {frame_width}x{frame_height} -> {new_width}x{new_height}")
        
        # Create temporary output file if needed
        temp_output = None
        if output_path is None:
            temp_fd, temp_output = tempfile.mkstemp(suffix='.mp4', dir=os.path.dirname(input_path))
            os.close(temp_fd)
            output_path = temp_output
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
        
        if not out.isOpened():
            logger.error("Cannot create output video writer")
            cap.release()
            return False
        
        # Reset to beginning and process all frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop frame
            cropped_frame = frame[start_y:end_y, 0:new_width]
            out.write(cropped_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        # Clean up
        cap.release()
        out.release()
        
        # Replace original file if needed
        if temp_output:
            os.replace(temp_output, input_path)
            logger.info(f"Overwrote original video: {input_path}")
        else:
            logger.info(f"Created cropped video: {output_path}")
        
        return True


def simple_crop_video_watermarks(video_path: str, 
                                bottom_percent: float = 10.0, 
                                top_percent: float = None,
                                use_smart_detection: bool = True) -> bool:
    """
    Simple function to crop watermarks from video.
    
    Args:
        video_path: Path to video file
        bottom_percent: Percentage of bottom to check/crop (if using simple mode)
        top_percent: Percentage of top to check/crop (if using simple mode)
        use_smart_detection: If True, use smart boundary detection. If False, use simple percentage.
        
    Returns:
        True if successful, False otherwise
    """
    cropper = SimpleWatermarkCropper()
    
    check_top = top_percent is not None
    check_bottom = bottom_percent is not None
    
    if use_smart_detection:
        return cropper.crop_video(video_path, check_top=check_top, check_bottom=check_bottom)
    else:
        # Use simple percentage mode
        simple_percent = bottom_percent if check_bottom else top_percent
        return cropper.crop_video(video_path, check_top=check_top, check_bottom=check_bottom, 
                                simple_percent=simple_percent)


# Example usage
if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"
    
    # Method 1: Simple - just crop bottom 10% if it's black
    success = simple_crop_video_watermarks(video_path, bottom_percent=10.0, use_smart_detection=False)
    print(f"Simple crop: {'successful' if success else 'failed'}")
    
    # Method 2: Smart detection (recommended)
    success = simple_crop_video_watermarks(video_path, bottom_percent=10.0, use_smart_detection=True)
    print(f"Smart crop: {'successful' if success else 'failed'}")
    
    # Method 3: Advanced usage
    cropper = SimpleWatermarkCropper(darkness_threshold=60, black_pixel_ratio=0.8)
    success = cropper.crop_video(video_path, check_bottom=True, simple_percent=12.0)
    print(f"Advanced crop: {'successful' if success else 'failed'}")