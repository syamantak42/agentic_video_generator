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
    - videos/{project}_2.mp4: Final composed video file

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
"""

from moviepy.editor import concatenate_videoclips, VideoFileClip
from proglog import ProgressBarLogger
import os
import sys
import re
import json
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


# -----------------------------
# Merge pre-generated clips
# -----------------------------
fps = 24
video_filename = project  + ".mp4"

clips_dir = os.path.join(project_root, "outputs", "clips")
if not os.path.isdir(clips_dir):
    raise FileNotFoundError(f"Clips directory not found: {clips_dir} - run generate_scripts.py first")

clip_files = [f for f in os.listdir(clips_dir) if f.lower().endswith('.mp4')]
clip_files = sorted(clip_files, key=extract_numbers)

print(f"Found {len(clip_files)} clips")
if not clip_files:
    raise RuntimeError(f"No clip MP4 files found in {clips_dir}")

video_clips = []
for index, fname in enumerate(clip_files, start=1):
    print(f"Loading final video clip {index}/{len(clip_files)}", flush=True)
    path = os.path.join(clips_dir, fname)
    video_clips.append(VideoFileClip(path))

print("\nMerging video clips...")
final_video = concatenate_videoclips(video_clips, method="compose")

output_path = os.path.join(video_dir, video_filename)
try:
    final_video.write_videofile(
        output_path,
        codec="libx264",
        fps=fps,
        logger=AppProgressLogger(),
    )
finally:
    final_video.close()
    for clip in video_clips:
        clip.close()

if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
    raise RuntimeError(f"Final video was not saved correctly: {output_path}")

print(f"\n[OK] Video generated successfully: {output_path}")



