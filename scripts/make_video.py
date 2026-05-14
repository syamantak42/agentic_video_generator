"""make_video.py

Module for composing final video from images and audio with animation effects.

Description:
    Creates final video by combining approved images with synchronized audio,
    applying Ken Burns or zoom animation effects to add visual dynamism.
    Generates high-resolution MP4 (2048x1152, 24fps) with automatic duration
    matching from audio files.

Inputs:
    - outputs/images/image_*.png: Approved images (sorted by filename)
    - outputs/audios/audio_*.mp3/wav: Audio segments (must match image count and order)

Outputs:
    - outputs/videos/{project}.mp4: Final composed video file

Configuration:
    - Output resolution: 2048x1152 pixels
    - Frame rate: 24 fps
    - Video codec: libx264 (H.264)
    - Animation: Zoom effect (adjustable to Ken Burns)

Usage:
    python make_video.py <project_name>
    
    Arguments:
        project_name: Project identifier (e.g., 'VikramBetaal')

Notes:
    - Image and audio files must be consistently ordered
    - Duration of each video segment matches corresponding audio
    - Intermediate .mp4 files created (one per image) and retained
    - An existing final video is archived with a timestamp suffix before composing
"""

import os
import sys
import re
import json
import subprocess
import time
from console_utils import configure_utf8_output

configure_utf8_output()

# --------------------------------------------------------
# Helper: Get project name and paths from config
# --------------------------------------------------------
def load_project_config(project_arg=None):
    """Load project name and TTS config from config.json or command-line arg."""
    project = project_arg or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not project:
        raise ValueError("Project name required: python script.py PROJECT_NAME")
    
    # Scripts are in PROJECT_NAME/scripts/, so go up one level to PROJECT_NAME/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.join(base_dir, project)
    source_dir = os.path.join(project_root, "source_material")
    config_path = os.path.join(source_dir, 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    project_name = config.get("_project_config", {}).get("project_name", project)
    return project_name, source_dir, config

# -----------------------------
# Load project name
# -----------------------------
project = sys.argv[1] if len(sys.argv) > 1 else None
project, source_dir, config = load_project_config(project)
project_root = os.path.dirname(source_dir)

# clips are assumed to already exist under this folder
clips_dir = os.path.join(project_root, "outputs", "clips")
print("clips directory:", clips_dir)

video_dir = os.path.join(project_root, "outputs", "videos")
os.makedirs(video_dir, exist_ok=True)
print("video directory:", video_dir)


# -----------------------------
# Helper: extract all numbers for proper sorting
# -----------------------------
def extract_numbers(name):
    """
    Extract all numeric sequences from filename for sorting.
    
    Args:
        name: Filename string
    
    Returns:
        tuple: Tuple of integers for consistent sorting (e.g., (1, 2) from 'image_1_2.png')
    
    Notes:
        Returns (0,) for filenames with no numbers
    """    
    nums = re.findall(r'\d+', name)
    return tuple(map(int, nums)) if nums else (0,)

def get_ffmpeg_binary():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def ffmpeg_concat_path(path):
    return os.path.abspath(path).replace("\\", "/").replace("'", "\\'")


def write_concat_list(clip_paths, list_path):
    with open(list_path, "w", encoding="utf-8") as f:
        for path in clip_paths:
            f.write(f"file '{ffmpeg_concat_path(path)}'\n")


def concat_with_ffmpeg_copy(clip_paths, output_path, list_path):
    write_concat_list(clip_paths, list_path)
    command = [
        get_ffmpeg_binary(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        output_path,
    ]

    print("Writing final video 1/2", flush=True)
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise RuntimeError(f"ffmpeg stream-copy concat could not start: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"ffmpeg stream-copy concat failed with exit code {result.returncode}.\n{stderr}")
    print("Writing final video 2/2", flush=True)


def concat_with_moviepy_reencode(clip_paths, output_path, fps):
    from moviepy.editor import concatenate_videoclips, VideoFileClip
    from proglog import ProgressBarLogger

    class AppProgressLogger(ProgressBarLogger):
        """Emit simple progress lines that Streamlit can render as a progress bar."""

        def callback(self, **changes):
            bars = self.state.get("bars", {})
            t_bar = bars.get("t")
            if not t_bar:
                return

            index = int(t_bar.get("index", 0))
            total = int(t_bar.get("total", 0) or 0)
            if total <= 0:
                return

            print(f"Writing final video {min(index, total)}/{total}", flush=True)

    video_clips = []
    for index, path in enumerate(clip_paths, start=1):
        print(f"Loading final video clip {index}/{len(clip_paths)}", flush=True)
        video_clips.append(VideoFileClip(path))

    print("\nMerging video clips with MoviePy fallback...")
    final_video = concatenate_videoclips(video_clips, method="compose")
    try:
        final_video.write_videofile(
            output_path,
            codec="libx264",
            fps=fps,
            logger=AppProgressLogger(),
            ffmpeg_params=["-preset", "ultrafast", "-threads", "0"],
        )
    finally:
        final_video.close()
        for clip in video_clips:
            clip.close()


def archive_existing_output(output_path):
    if not os.path.exists(output_path):
        return None

    directory = os.path.dirname(output_path)
    stem, ext = os.path.splitext(os.path.basename(output_path))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(directory, f"{stem}_previous_{timestamp}{ext}")
    counter = 2
    while os.path.exists(backup_path):
        backup_path = os.path.join(directory, f"{stem}_previous_{timestamp}_{counter}{ext}")
        counter += 1

    os.replace(output_path, backup_path)
    print(f"COMPOSE_FINAL_VIDEO_ARCHIVED_OLD: {backup_path}", flush=True)
    return backup_path


# -----------------------------
# Merge pre-generated clips
# -----------------------------
fps = 24
output_project_name = os.path.basename(project_root)
video_filename = output_project_name + ".mp4"

clips_dir = os.path.join(project_root, "outputs", "clips")
if not os.path.isdir(clips_dir):
    raise FileNotFoundError(f"Clips directory not found: {clips_dir} - run generate_scripts.py first")

clip_files = [f for f in os.listdir(clips_dir) if f.lower().endswith('.mp4')]
clip_files = sorted(clip_files, key=extract_numbers)

print(f"Found {len(clip_files)} clips")
if not clip_files:
    raise RuntimeError(f"No clip MP4 files found in {clips_dir}")

output_path = os.path.join(video_dir, video_filename)
archive_existing_output(output_path)
temp_output_path = os.path.join(video_dir, f"_{output_project_name}_compose_tmp.mp4")
list_path = os.path.join(video_dir, "_clips_concat_list.txt")
clip_paths = [os.path.join(clips_dir, fname) for fname in clip_files]

if os.path.exists(temp_output_path):
    os.remove(temp_output_path)

try:
    print("\nMerging video clips with ffmpeg stream copy...")
    concat_with_ffmpeg_copy(clip_paths, temp_output_path, list_path)
except RuntimeError as exc:
    print(f"[WARN] Fast ffmpeg concat failed. Falling back to MoviePy re-encode.\n{exc}", flush=True)
    if os.path.exists(temp_output_path):
        try:
            os.remove(temp_output_path)
        except OSError:
            pass
    concat_with_moviepy_reencode(clip_paths, temp_output_path, fps)
finally:
    if os.path.exists(list_path):
        try:
            os.remove(list_path)
        except OSError:
            pass

if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) <= 0:
    raise RuntimeError(f"Final video was not composed correctly: {temp_output_path}")

os.replace(temp_output_path, output_path)

if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
    raise RuntimeError(f"Final video was not saved correctly: {output_path}")

print(f"COMPOSE_FINAL_VIDEO_SIZE: {os.path.getsize(output_path)}", flush=True)
print(f"COMPOSE_FINAL_VIDEO_OK: {output_path}", flush=True)
print(f"\n[OK] Video generated successfully: {output_path}")



