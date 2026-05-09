"""generate_scripts.py

Module for creating animated video clips with synchronized audio.

Description:
    Processes approved images and corresponding audio segments to produce
    individual MP4 clips. Each clip applies a zoom animation (or Ken Burns
    effect) to the image for the duration of its audio, then embeds the audio.
    Generated clips are saved under `outputs/clips` and may later be
    stitched together by `make_video.py`.

Inputs:
    - outputs/images/image_*.png
    - outputs/audios/audio_*.mp3 or .wav

Outputs:
    - outputs/clips/image_*.mp4  (video+audio)

Usage:
    python generate_scripts.py <project_name>

    Arguments:
        project_name: Project identifier (e.g., 'VikramBetaal')
"""

import os
import sys
import json
import re
import numpy as np
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip
from pydub import AudioSegment
from console_utils import configure_utf8_output

configure_utf8_output()


# --------------------------------------------------------
# Helper: Get project name and paths from config
# --------------------------------------------------------
def load_project_config(project_arg=None):
    """Load project name and configuration from config.json.

    project_arg may be provided on the command line; if omitted the first
    positional argument is used. If still missing an error is raised.
    """
    project = project_arg or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not project:
        raise ValueError("Project name required: python script.py PROJECT_NAME")

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


# ---------------------------------------------
# Main
# ---------------------------------------------

project = sys.argv[1] if len(sys.argv) > 1 else None
project, source_dir, config = load_project_config(project)
project_root = os.path.dirname(source_dir)

# determine audio folder using tts model from config
tts_model = config.get("_project_config", {}).get("tts_config", {}).get("model", "kokoro").lower()
audio_dir = os.path.join(project_root, "outputs", "audios")
print("audio directory:", audio_dir)

image_dir = os.path.join(project_root, "outputs", "images")
print("image directory:", image_dir)

clips_dir = os.path.join(project_root, "outputs", "clips")
os.makedirs(clips_dir, exist_ok=True)
print("clips directory:", clips_dir)

# helper for sorting filenames by contained numbers
def extract_numbers(name):
    nums = re.findall(r'\d+', name)
    return tuple(map(int, nums)) if nums else (0,)

# gather and sort files
def sorted_files(dirpath, extensions):
    files = [
        f for f in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, f))
        and f.lower().startswith("image_")
        and any(f.lower().endswith(ext) for ext in extensions)
    ]
    return sorted(files, key=extract_numbers)

image_paths = sorted_files(image_dir, ['.png'])
audio_paths = sorted_files(audio_dir, ['.mp3', '.wav'])

if len(image_paths) != len(audio_paths):
    print("Warning: number of images and audio files differ")

# compute durations using pydub
fps = 24

durations = []
for filename in audio_paths:
    file_path = os.path.join(audio_dir, filename)
    audio = AudioSegment.from_file(file_path)
    duration_sec = len(audio) / 1000
    durations.append(duration_sec)

# animation helpers

def zoom_effect(image_path, duration, fps=24):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    zoom_levels = np.linspace(1, 1.4, num=int(duration * fps))

    frames = []
    for zoom in zoom_levels:
        center = (w // 2, h // 2)
        size = (int(w / zoom), int(h / zoom))
        cropped = img[
            center[1] - size[1] // 2 : center[1] + size[1] // 2,
            center[0] - size[0] // 2 : center[0] + size[0] // 2
        ]
        resized = cv2.resize(cropped, (w, h))
        frames.append(resized)
    return frames


def ken_burnes(image_path, duration, fps=24):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    start_x, start_y = 0, 0
    end_x, end_y = int(w * 0.2), int(h * 0.2)
    num_frames = int(duration * fps)
    zoom_levels = np.linspace(1, 1.3, num=num_frames)
    x_offsets = np.linspace(start_x, end_x, num=num_frames)
    y_offsets = np.linspace(start_y, end_y, num=num_frames)

    frames = []
    for zoom, x_offset, y_offset in zip(zoom_levels, x_offsets, y_offsets):
        crop_w, crop_h = int(w / zoom), int(h / zoom)
        x = x_offset + (w - crop_w) // 2
        y = y_offset + (h - crop_h) // 2
        x = max(0, min(x, w - crop_w))
        y = max(0, min(y, h - crop_h))
        cropped = img[int(y):int(y + crop_h), int(x):int(x + crop_w)]
        resized = cv2.resize(cropped, (w, h))
        frames.append(resized)
    return frames

# create clips
for i, img in enumerate(image_paths):
    img_path = os.path.join(image_dir, img)
    duration = durations[i] if i < len(durations) else None
    if duration is None:
        print(f"Skipping {img}: no corresponding audio")
        continue

    # choose animation effect here; currently zoom_effect
    frames = zoom_effect(img_path, duration, fps)

    clip = ImageSequenceClip(frames, fps=fps)
    audio_path = os.path.join(audio_dir, audio_paths[i])
    audio_clip = AudioFileClip(audio_path).set_duration(duration)
    clip = clip.set_audio(audio_clip)

    clip_path = os.path.join(clips_dir, f"{os.path.splitext(img)[0]}.mp4")
    clip.write_videofile(clip_path, codec="libx264", fps=fps)
    print(f"Generated clip: {clip_path}")

print("\n[OK] All clips created.")



