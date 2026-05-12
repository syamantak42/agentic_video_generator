"""generate_inworld_voice.py

Module for generating text-to-speech audio using Inworld AI API.

Description:
    Converts narration text segments into MP3 audio files using Inworld TTS API.
    Processes a narration JSON file with multiple sections and creates individual
    audio files for each narration segment with consistent voice settings.

Inputs:
    - outputs/output_jsons/narration.json: Structured narration with sections and narration_text segments

Outputs:
    - outputs/audios/audio_<section>_<segment>.mp3: Generated MP3 audio files
    - outputs/output_jsons/tts_index.json: Metadata index mapping audio files to narration segments

Environment Variables:
    - INWORLD_API_KEY: API key for Inworld TTS service (Base64 encoded if required)

Configuration:
    - VOICE_ID: selected in source_material/config.json
    - MODEL_ID: selected in source_material/config.json
    - TEMPERATURE: 1.0 (vocal variation/expressiveness)
    - SPEAKING_RATE: 1.0 (normal speed)

Usage:
    python generate_inworld_voice.py <project_name>
    
    Arguments:
        project_name: Project identifier (e.g., 'VikramBetaal')
"""
from dotenv import load_dotenv
import os
import json
import base64
import requests
import sys
import argparse
import re
from console_utils import configure_utf8_output

configure_utf8_output()

# --------------------------------------------------------
# Helper: Get project name and paths from config
# --------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Inworld TTS audio from narration.json.")
    parser.add_argument("project", help="Project name")
    parser.add_argument(
        "--audio-selection",
        default="all",
        help=(
            "Audio selection: 'all', 'missing', a single key like '1_2', "
            "a comma list like '1_1,2_3', or a range like '1_1-3_2'."
        ),
    )
    return parser.parse_args()


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

args = parse_args()
project, source_dir, config = load_project_config(args.project)

# derive base directories relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(source_dir)

INPUT_JSON = os.path.join(project_root, "outputs", "output_jsons", "narration.json")
OUTPUT_DIR = os.path.join(project_root, "outputs", "audios")

# ---------------- Configuration ----------------

tts_config = config.get("_project_config", {}).get("tts_config", {})
VOICE_ID = (tts_config.get("voice_id", "Ashley") or "Ashley").strip()
MODEL_ID = (tts_config.get("model", "inworld-tts-1.5-max") or "inworld-tts-1.5-max").strip()
if MODEL_ID == "inworld":
    MODEL_ID = "inworld-tts-1.5-max"
TEMPERATURE = 1.0
SPEAKING_RATE = 1.0

# ------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
ENV_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), ".env")
load_dotenv(dotenv_path=ENV_PATH)
api_key = os.getenv("INWORLD_API_KEY")


if not api_key:
    raise ValueError("Missing INWORLD_API_KEY environment variable")

URL = "https://api.inworld.ai/tts/v1/voice"
HEADERS = {
    "Authorization": f"Basic {api_key}",
    "Content-Type": "application/json"
}


def synthesize_tts(text, out_path):
    """
    Synthesize text to speech and save as MP3 using Inworld API.
    
    Args:
        text: Narration text to convert to audio
        out_path: File path where MP3 will be saved
    
    Returns:
        str: Path to saved MP3 file
    
    Raises:
        requests.HTTPError: If API request fails
    """
    payload = {
        "text": text,
        "voiceId": VOICE_ID,
        "modelId": MODEL_ID,
        "temperature": TEMPERATURE,
        "speakingRate": SPEAKING_RATE
    }

    resp = requests.post(URL, json=payload, headers=HEADERS)
    resp.raise_for_status()
    result = resp.json()
    audio_data = base64.b64decode(result["audioContent"])

    with open(out_path, "wb") as f:
        f.write(audio_data)
    return out_path


def format_key(key):
    return f"{key[0]}_{key[1]}"


def parse_audio_key(value):
    match = re.match(r"^\s*(\d+)[_\-:](\d+)\s*$", value)
    if not match:
        raise ValueError(f"Invalid audio key: {value}. Use section_segment, e.g. 3_2.")
    return int(match.group(1)), int(match.group(2))


def existing_audio_keys(output_dir):
    keys = set()
    if not os.path.isdir(output_dir):
        return keys
    for filename in os.listdir(output_dir):
        if not filename.lower().endswith((".mp3", ".wav")):
            continue
        stem = os.path.splitext(filename)[0]
        match = re.match(r"^audio_(\d+)_(\d+)$", stem, flags=re.IGNORECASE)
        if match:
            keys.add((int(match.group(1)), int(match.group(2))))
    return keys


def select_audio_keys(selection, available_keys, output_dir):
    selection = (selection or "all").strip().lower()
    available = sorted(available_keys)

    if selection == "all":
        return available

    if selection == "missing":
        return sorted(set(available) - existing_audio_keys(output_dir))

    if "-" in selection and "," not in selection:
        start_text, end_text = selection.split("-", 1)
        start_key = parse_audio_key(start_text)
        end_key = parse_audio_key(end_text)
        if start_key > end_key:
            start_key, end_key = end_key, start_key
        return [key for key in available if start_key <= key <= end_key]

    requested = [parse_audio_key(part) for part in selection.split(",") if part.strip()]
    missing_requested = sorted(set(requested) - set(available))
    if missing_requested:
        missing_text = ", ".join(format_key(key) for key in missing_requested)
        raise ValueError(f"Requested audio frames are not in narration.json: {missing_text}")
    return sorted(dict.fromkeys(requested))


def find_audio_file(output_dir, key):
    section, segment = key
    preferred = [
        os.path.join(output_dir, f"audio_{section}_{segment}.mp3"),
        os.path.join(output_dir, f"audio_{section}_{segment}.wav"),
    ]
    for path in preferred:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return ""


def main():
    """
    Main execution function: process narration sections and generate audio files.
    
    Reads narration.json, generates TTS audio for each segment, and creates
    an index file mapping audio files to narration metadata.
    """
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_title = data.get("video_title", "untitled_video")
    sections = data.get("sections", [])

    segments_by_key = {}

    for si, section in enumerate(sections, 1):
        title = section["section_title"]
        texts = section.get("narration_text", [])

        for ti, segment in enumerate(texts, 1):
            segment_text = segment.strip()
            if not segment_text:
                continue
            segments_by_key[(si, ti)] = {
                "title": title,
                "text": segment_text,
            }

    selected_keys = select_audio_keys(args.audio_selection, set(segments_by_key), OUTPUT_DIR)
    if not selected_keys:
        print(f"No audio selected for generation ({args.audio_selection}).")

    generated_count = 0
    for index, key in enumerate(selected_keys, start=1):
        si, ti = key
        title = segments_by_key[key]["title"]
        segment_text = segments_by_key[key]["text"]

        # -----------------------------------------
        # UPDATED NAMING SCHEME:
        # audio_<section>_<segment>.mp3
        # -----------------------------------------
        filename = f"audio_{si}_{ti}.mp3"
        out_path = os.path.join(OUTPUT_DIR, filename)

        print(f"Generating audio {index}/{len(selected_keys)}", flush=True)
        print(f"[TTS] {video_title} | Section {si}.{ti}: {title}")
        synthesize_tts(segment_text, out_path)
        generated_count += 1

    index = []
    for key in sorted(segments_by_key):
        audio_file = find_audio_file(OUTPUT_DIR, key)
        if not audio_file:
            continue
        si, ti = key
        index.append({
            "section": si,
            "segment": ti,
            "title": segments_by_key[key]["title"],
            "text": segments_by_key[key]["text"],
            "audio_file": audio_file
        })

    index_path = os.path.join(project_root, "outputs", "output_jsons", "tts_index.json")
    with open(index_path, "w", encoding="utf-8") as jf:
        json.dump(index, jf, ensure_ascii=False, indent=2)

    print(f"\n[OK] Generated {generated_count} audio files. Indexed {len(index)} audio files.")
    print(f"Index saved to: {index_path}")


if __name__ == "__main__":
    main()
