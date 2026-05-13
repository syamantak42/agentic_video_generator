"""generate_clips.py

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
    python generate_clips.py <project_name>

    Arguments:
        project_name: Project identifier (e.g., 'VikramBetaal')
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from pydub import AudioSegment

from console_utils import configure_utf8_output

configure_utf8_output()

IMAGE_EXTENSIONS = {".png"}
AUDIO_EXTENSIONS = {".mp3", ".wav"}
TEMP_FILE_PATTERN = re.compile(r".*TEMP_MPY_wvf_snd\.(mp3|mp4|m4a|wav)$", re.IGNORECASE)
CLIP_KEY_PATTERN = re.compile(r"^\s*(\d+)[_\-:](\d+)\s*$")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video clips from approved images and audio.")
    parser.add_argument("project", help="Project name")
    parser.add_argument(
        "--clip-selection",
        default="all",
        help=(
            "Clip selection: 'all', 'missing', a single key like '1_2', "
            "a comma list like '1_1,2_3', or a range like '1_1-3_2'."
        ),
    )
    return parser.parse_args()


def load_project_config(project_arg=None):
    project = project_arg or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not project:
        raise ValueError("Project name required: python generate_clips.py PROJECT_NAME")

    base_dir = Path(__file__).resolve().parents[1]
    project_root = base_dir / project
    source_dir = project_root / "source_material"
    config_path = source_dir / "config.json"

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    project_name = config.get("_project_config", {}).get("project_name", project)
    return project_name, source_dir, config


def parse_media_key(filename, prefix):
    stem = Path(filename).stem
    match = re.match(rf"^{re.escape(prefix)}_(\d+)_(\d+)$", stem, flags=re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else None


def collect_media_files(dirpath, prefix, extensions):
    directory = Path(dirpath)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files_by_key = {}
    for entry in directory.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in extensions:
            continue

        key = parse_media_key(entry.name, prefix)
        if key is None:
            continue
        if key in files_by_key:
            raise ValueError(
                f"Duplicate {prefix} file for section {key[0]}, segment {key[1]}: "
                f"{files_by_key[key]} and {entry.name}"
            )
        files_by_key[key] = entry.name

    return files_by_key


def format_key(key):
    return f"{key[0]}_{key[1]}"


def parse_clip_key(value):
    match = CLIP_KEY_PATTERN.match(value)
    if not match:
        raise ValueError(f"Invalid clip key: {value}. Use section_segment, e.g. 3_2.")
    return int(match.group(1)), int(match.group(2))


def select_clip_keys(selection, available_keys, clips_dir):
    selection = (selection or "all").strip().lower()
    available = sorted(available_keys)

    if selection == "all":
        return available

    if selection == "missing":
        existing_clips = collect_media_files(clips_dir, "image", {".mp4"})
        return sorted(set(available) - set(existing_clips))

    if "-" in selection and "," not in selection:
        start_text, end_text = selection.split("-", 1)
        start_key = parse_clip_key(start_text)
        end_key = parse_clip_key(end_text)
        if start_key > end_key:
            start_key, end_key = end_key, start_key
        return [key for key in available if start_key <= key <= end_key]

    requested = [parse_clip_key(part) for part in selection.split(",") if part.strip()]
    missing_requested = sorted(set(requested) - set(available))
    if missing_requested:
        missing_text = ", ".join(format_key(key) for key in missing_requested)
        raise ValueError(f"Requested clips do not have matching image/audio pairs: {missing_text}")
    return sorted(dict.fromkeys(requested))


def cleanup_moviepy_temp_files(*directories):
    for directory in directories:
        path = Path(directory)
        if not path.is_dir():
            continue
        for entry in path.iterdir():
            if entry.is_file() and TEMP_FILE_PATTERN.match(entry.name):
                try:
                    entry.unlink()
                except OSError:
                    pass


def get_audio_duration(audio_path):
    audio = AudioSegment.from_file(str(audio_path))
    return len(audio) / 1000.0


def get_ffmpeg_binary():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def build_zoom_filter(fps, duration):
    frame_count = max(1, int(round(duration * fps)))
    return (
        "scale=8192:4608:force_original_aspect_ratio=increase,"
        "crop=8192:4608,"
        f"zoompan=z='1.0+0.4*on/{frame_count}':"
        "x='(iw/2)-(ow/2)':"
        "y='(ih/2)-(oh/2)':"
        f"d={frame_count}:s=2048x1152:fps={fps},"
        "format=yuv420p"
    )


def render_zoom_clip(image_path, audio_path, duration, output_path, fps=24):
    image_path = Path(image_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not image_path.is_file():
        raise RuntimeError(f"Image file not found: {image_path}")
    if not audio_path.is_file():
        raise RuntimeError(f"Audio file not found: {audio_path}")

    zoom_filter = build_zoom_filter(fps, duration)
    filter_complex = f"[0:v]{zoom_filter}[v]"
    command = [
        get_ffmpeg_binary(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        str(image_path),
        "-i",
        str(audio_path),
        "-t",
        f"{duration:.3f}",
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-threads",
        "2",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg failed with exit code {result.returncode} while writing {output_path}.\n{stderr.strip()}"
        )


def main():
    args = parse_args()
    project, source_dir, config = load_project_config(args.project)
    project_root = source_dir.parent
    audio_dir = project_root / "outputs" / "audios"
    image_dir = project_root / "outputs" / "images"
    clips_dir = project_root / "outputs" / "clips"
    clip_temp_dir = clips_dir / "_temp"

    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_temp_dir.mkdir(parents=True, exist_ok=True)

    cleanup_moviepy_temp_files(Path.cwd(), clip_temp_dir)

    image_files = collect_media_files(image_dir, "image", IMAGE_EXTENSIONS)
    audio_files = collect_media_files(audio_dir, "audio", AUDIO_EXTENSIONS)

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

    media_keys = select_clip_keys(args.clip_selection, sorted(image_keys), clips_dir)
    if not media_keys:
        print(f"No clips selected for generation ({args.clip_selection}).")
        return

    print(f"Found {len(image_keys)} matched image/audio pairs")
    print(f"Selected {len(media_keys)} clips for generation: {', '.join(format_key(key) for key in media_keys)}")

    fps = 24
    durations = {key: get_audio_duration(audio_dir / audio_files[key]) for key in media_keys}

    created_count = 0
    failed_clips = []

    for index, key in enumerate(media_keys, start=1):
        img_name = image_files[key]
        audio_name = audio_files[key]
        img_path = image_dir / img_name
        audio_path = audio_dir / audio_name
        duration = durations[key]
        temp_clip_path = clip_temp_dir / f"{Path(img_name).stem}_rendering.mp4"
        clip_path = clips_dir / f"{Path(img_name).stem}.mp4"

        try:
            print(f"Generating clip {index}/{len(media_keys)}", flush=True)
            if temp_clip_path.exists():
                temp_clip_path.unlink()

            render_zoom_clip(img_path, audio_path, duration, temp_clip_path, fps=fps)

            if not temp_clip_path.exists() or temp_clip_path.stat().st_size <= 0:
                raise RuntimeError(f"Temporary clip was not written correctly: {temp_clip_path}")

            temp_clip_path.replace(clip_path)
            if not clip_path.exists() or clip_path.stat().st_size <= 0:
                raise RuntimeError(f"Final clip was not saved correctly: {clip_path}")

            created_count += 1
            print(f"Generated clip: {clip_path}")
        except Exception as exc:
            failed_clips.append((format_key(key), str(exc)))
            print(f"[ERROR] Failed clip {format_key(key)}: {exc}")
        finally:
            if temp_clip_path.exists():
                try:
                    temp_clip_path.unlink()
                except OSError:
                    pass

    if failed_clips:
        failed_keys = ", ".join(key for key, _ in failed_clips)
        raise RuntimeError(f"Failed to create {len(failed_clips)} clips: {failed_keys}")

    if created_count != len(media_keys):
        raise RuntimeError(f"Created {created_count}/{len(media_keys)} clips.")

    print(f"\n[OK] Created {created_count}/{len(media_keys)} clips.")
    cleanup_moviepy_temp_files(Path.cwd(), clip_temp_dir)


if __name__ == "__main__":
    main()



