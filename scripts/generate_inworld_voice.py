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
    - VOICE_ID: 'Hades' (voice character)
    - MODEL_ID: 'inworld-tts-1-max' (highest quality model)
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

project = sys.argv[1] if len(sys.argv) > 1 else None
project, source_dir, config = load_project_config(project)

# derive base directories relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(source_dir)

INPUT_JSON = os.path.join(project_root, "outputs", "output_jsons", "narration.json")
OUTPUT_DIR = os.path.join(project_root, "outputs", "audios")

# ---------------- Configuration ----------------

tts_config = config.get("_project_config", {}).get("tts_config", {})
VOICE_ID = tts_config.get("voice_id", "Ashley")
MODEL_ID = "inworld-tts-1-max"
TEMPERATURE = 1.0
SPEAKING_RATE = 0.8

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

    index = []  # metadata for reference

    for si, section in enumerate(sections, 1):
        title = section["section_title"]
        texts = section.get("narration_text", [])

        for ti, segment in enumerate(texts, 1):
                segment_text = segment.strip()
                if not segment_text:
                    continue

                # -----------------------------------------
                # UPDATED NAMING SCHEME:
                # audio_<section>_<segment>.mp3
                # -----------------------------------------
                filename = f"audio_{si}_{ti}.mp3"
                out_path = os.path.join(OUTPUT_DIR, filename)

                print(f"[TTS] {video_title} | Section {si}.{ti}: {title}")
                synthesize_tts(segment_text, out_path)

                index.append({
                    "section": si,
                    "segment": ti,
                    "title": title,
                    "text": segment_text,
                    "audio_file": out_path
                })

    index_path = os.path.join(project_root, "outputs", "output_jsons", "tts_index.json")
    with open(index_path, "w", encoding="utf-8") as jf:
        json.dump(index, jf, ensure_ascii=False, indent=2)

    print(f"\n[OK] Generated {len(index)} audio files.")
    print(f"Index saved to: {index_path}")


if __name__ == "__main__":
    main()



