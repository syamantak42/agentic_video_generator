"""generate_scripts.py

Module for creating animated video clips with synchronized audio.

Description:
    Processes approved images and corresponding audio segments to produce
    individual MP4 clips. Each clip applies a zoom animation (or Ken Burns
    effect) to the image for the duration of its audio, then embeds the audio.
    Generated clips are saved under `outputs/clips` and may later be
    stitched together by `make_video.py`.

Inputs:
    - outputs/images/image_<section>_<segment>.png
    - outputs/audios/audio_<section>_<segment>.mp3 or .wav

Outputs:
    - outputs/clips/image_<section>_<segment>.mp4  (video+audio)

Usage:
    python generate_scripts.py <project_name>

    Arguments:
        project_name: Project identifier (e.g., 'VikramBetaal')
"""

import os
import sys
import json
import re
import math
import subprocess
import cv2
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

clip_temp_dir = os.path.join(clips_dir, "_temp")
os.makedirs(clip_temp_dir, exist_ok=True)

def parse_media_key(filename, prefix):
    """Return (section, segment) for names like image_1_2.png or audio_1_2.mp3."""
    stem = os.path.splitext(filename)[0]
    match = re.match(rf"^{re.escape(prefix)}_(\d+)_(\d+)$", stem, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def collect_media_files(dirpath, prefix, extensions):
    """Collect media files by section/segment key."""
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"Directory not found: {dirpath}")

    files_by_key = {}
    for filename in os.listdir(dirpath):
        path = os.path.join(dirpath, filename)
        if not os.path.isfile(path):
            continue
        if not any(filename.lower().endswith(ext) for ext in extensions):
            continue

        key = parse_media_key(filename, prefix)
        if key is None:
            continue
        if key in files_by_key:
            raise ValueError(
                f"Duplicate {prefix} file for section {key[0]}, segment {key[1]}: "
                f"{files_by_key[key]} and {filename}"
            )
        files_by_key[key] = filename

    return files_by_key


def format_key(key):
    return f"{key[0]}_{key[1]}"


def cleanup_moviepy_temp_files(*directories):
    """Remove stale MoviePy temp audio files created by interrupted runs."""
    temp_pattern = re.compile(r".*TEMP_MPY_wvf_snd\.(mp3|mp4|m4a|wav)$", re.IGNORECASE)
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if not temp_pattern.match(filename):
                continue
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


cleanup_moviepy_temp_files(os.getcwd(), clip_temp_dir)


image_files = collect_media_files(image_dir, "image", [".png"])
audio_files = collect_media_files(audio_dir, "audio", [".mp3", ".wav"])

if not image_files:
    raise RuntimeError(f"No approved image files found in {image_dir}")
if not audio_files:
    raise RuntimeError(f"No audio files found in {audio_dir}")

image_keys = set(image_files)
audio_keys = set(audio_files)
missing_audio = sorted(image_keys - audio_keys)
missing_images = sorted(audio_keys - image_keys)

if missing_audio or missing_images:
    if missing_audio:
        print("Missing audio for image keys:", ", ".join(format_key(key) for key in missing_audio))
    if missing_images:
        print("Missing images for audio keys:", ", ".join(format_key(key) for key in missing_images))
    raise RuntimeError(
        f"Image/audio mismatch: {len(image_files)} images, {len(audio_files)} audio files. "
        "Expected matching image_<section>_<segment> and audio_<section>_<segment> files."
    )

media_keys = sorted(image_keys)
print(f"Found {len(media_keys)} matched image/audio pairs")

# compute durations using pydub
fps = 24

durations = []
for key in media_keys:
    filename = audio_files[key]
    file_path = os.path.join(audio_dir, filename)
    audio = AudioSegment.from_file(file_path)
    duration_sec = len(audio) / 1000
    durations.append(duration_sec)

# animation helpers

def get_ffmpeg_binary():
    """Use the ffmpeg bundled with imageio/moviepy when available."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def make_zoom_frame(img, t, duration, zoom_start=1.0, zoom_end=1.4):
    """Create one RGB zoom frame from an already-loaded image."""
    h, w, _ = img.shape
    progress = min(1.0, max(0.0, t / max(duration, 0.001)))
    zoom = zoom_start + (zoom_end - zoom_start) * progress
    center = (w // 2, h // 2)
    size = (int(w / zoom), int(h / zoom))
    cropped = img[
        center[1] - size[1] // 2 : center[1] + size[1] // 2,
        center[0] - size[0] // 2 : center[0] + size[0] // 2
    ]
    return cv2.resize(cropped, (w, h))


def render_zoom_clip(image_path, audio_path, duration, output_path, fps=24):
    """Stream generated RGB frames into ffmpeg and mux them with audio."""
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image file: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    frame_count = max(1, math.ceil(duration * fps))

    command = [
        get_ffmpeg_binary(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        output_path,
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        assert process.stdin is not None
        for frame_index in range(frame_count):
            frame = make_zoom_frame(img, frame_index / fps, duration)
            process.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass
    finally:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass

    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {return_code} while writing {output_path}.\n{stderr.strip()}"
        )

# create clips
created_count = 0
failed_clips = []

for i, key in enumerate(media_keys):
    img = image_files[key]
    audio = audio_files[key]
    img_path = os.path.join(image_dir, img)
    audio_path = os.path.join(audio_dir, audio)
    duration = durations[i]
    clip_path = os.path.join(clips_dir, f"{os.path.splitext(img)[0]}.mp4")
    temp_clip_path = os.path.join(clip_temp_dir, f"{os.path.splitext(img)[0]}_rendering.mp4")

    try:
        print(f"Generating clip {i + 1}/{len(media_keys)}", flush=True)
        if os.path.exists(temp_clip_path):
            os.remove(temp_clip_path)

        render_zoom_clip(
            img_path,
            audio_path,
            duration,
            temp_clip_path,
            fps=fps,
        )
        if not os.path.exists(temp_clip_path) or os.path.getsize(temp_clip_path) <= 0:
            raise RuntimeError(f"Temporary clip was not written correctly: {temp_clip_path}")

        os.replace(temp_clip_path, clip_path)
        if not os.path.exists(clip_path) or os.path.getsize(clip_path) <= 0:
            raise RuntimeError(f"Final clip was not saved correctly: {clip_path}")

        created_count += 1
        print(f"Generated clip: {clip_path}")
    except Exception as exc:
        failed_clips.append((format_key(key), str(exc)))
        print(f"[ERROR] Failed clip {format_key(key)}: {exc}")
    finally:
        if os.path.exists(temp_clip_path):
            try:
                os.remove(temp_clip_path)
            except OSError:
                pass
        cleanup_moviepy_temp_files(clip_temp_dir)

if failed_clips:
    failed_keys = ", ".join(key for key, _ in failed_clips)
    raise RuntimeError(f"Failed to create {len(failed_clips)} clips: {failed_keys}")

if created_count != len(media_keys):
    raise RuntimeError(f"Created {created_count}/{len(media_keys)} clips.")

print(f"\n[OK] Created {created_count}/{len(media_keys)} clips.")
cleanup_moviepy_temp_files(os.getcwd(), clip_temp_dir)



