import os
import yt_dlp


def download_video(video_url, filename, root):
    os.makedirs(root, exist_ok=True)

    output_path = os.path.join(root, f"{filename}.mp4")

    # Specify a more compatible format - exclude AV1 codec
    # Preferring h264 codec which is widely supported
    ydl_opts = {
        "format": "bestvideo[ext=mp4][vcodec!*=av01][vcodec!*=av1]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!*=av01][vcodec!*=av1]/best",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        # Add cookie authentication
        "extractor_retries": 3,
        "fragment_retries": 3,
        "sleep_interval": 1,
        "max_sleep_interval": 5,

        # Add postprocessors to ensure compatible format
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
        # Force a specific codec if needed
        "postprocessor_args": [
            "-vcodec",
            "libx264",  # Force H.264 codec
            "-acodec",
            "aac",  # Force AAC audio codec
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Attempting to download: {video_url}")
            print(f"Saving path: {output_path}")
            ydl.download([video_url])

        if os.path.exists(output_path):
            print(f"Downloaded: {output_path}; {video_url}")
            return output_path, True
        else:
            print(
                f"Failed to download {video_url}. File not found at expected location."
            )
            return None, False

    except Exception as e:
        print(f"Exception during download of {video_url}: {e}")
        return None, False


def main():
    ROOT_DIR = "/BRAIN/adv-robustness/work/video_hall/mvp/assets/clips"

    video_url = "https://www.youtube.com/watch?v=Axru07JeBig"
    video_title = video_url.split("?v=")[-1]

    try:
        vid_path = f"{ROOT_DIR}/{video_title}.mp4"
        if not os.path.isfile(vid_path):
            video_path, did_download = download_video(
                video_url,
                f"{video_title}",
                root=ROOT_DIR,
            )
            assert video_path is not None
            assert did_download
        else:
            print(f"Video Already Exists!")
            video_path = vid_path
        assert video_path is not None
    except Exception as e:
        print(f"Got Exception: {e}")


if __name__ == "__main__":
    main()