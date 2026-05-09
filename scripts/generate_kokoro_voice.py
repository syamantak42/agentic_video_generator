import os
import sys
import json
import base64
import time
import numpy as np
import soundfile as sf
import subprocess
from kokoro import KPipeline

# ------------------------------------------------
# Helper: Get project name and paths from config
# ------------------------------------------------
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

# ------------------------------------------------
# Configuration
# ------------------------------------------------
SLOWNESS = 0.9  # slow audio factor

# ------------------------------------------------
# Inputs identical to inworld script
# ------------------------------------------------

project = sys.argv[1] if len(sys.argv) > 1 else None
project, source_dir, config = load_project_config(project)
tts_config = config.get("_project_config", {}).get("tts_config", {})

# base directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(source_dir)

INPUT_JSON = os.path.join(project_root, "outputs", "output_jsons", "narration.json")

OUTPUT_DIR = os.path.join(project_root, "outputs", "audios")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# kokoro TTS
pipeline = KPipeline(lang_code='a')


def normalize_kokoro_voice(voice_id):
    """Accept short ids like 'af' and full ids like 'af_heart'."""
    if not voice_id:
        return "af_heart"
    return voice_id if "_" in voice_id else f"{voice_id}_heart"


KOKORO_VOICE = normalize_kokoro_voice(tts_config.get("voice_id", "af_heart"))


def slow_down_audio(input_path, output_path, speed):
    subprocess.run([
        'ffmpeg', '-y', '-i', input_path,
        '-filter:a', f"atempo={speed}",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def synthesize_kokoro(text, tmp_path, final_path, slow_factor):
    # Generate original speed WAV first
    generator = pipeline(text, voice=KOKORO_VOICE)
    chunks = [audio for _, _, audio in generator]
    full_audio = np.concatenate(chunks)
    sf.write(tmp_path, full_audio, 24000)

    # Convert to slow version (final output)
    slow_down_audio(tmp_path, final_path, speed=slow_factor)

    # Remove temporary fast file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return final_path


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_title = data.get("video_title", "untitled_video")
    sections = data.get("sections", [])

    index = []

    for si, section in enumerate(sections, 1):
        title = section["section_title"]
        texts = section.get("narration_text", [])

        for ti, segment in enumerate(texts, 1):
            segment_text = segment.strip()
            if not segment_text:
                continue

            print(f"[KOKORO] {video_title} | Section {si}.{ti}: {title}")

            # final filename (slow version)
            filename = f"audio_{si}_{ti}.wav"
            final_path = os.path.join(OUTPUT_DIR, filename)

            # tmp file for initial TTS
            tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{si}_{ti}.wav")

            synthesize_kokoro(segment_text, tmp_path, final_path, SLOWNESS)

            index.append({
                "section": si,
                "segment": ti,
                "title": title,
                "text": segment_text,
                "audio_file": final_path
            })

    index_path = os.path.join(project_root, "outputs", "output_jsons", "tts_index.json")
    with open(index_path, "w", encoding="utf-8") as jf:
        json.dump(index, jf, ensure_ascii=False, indent=2)

    print(f"\n[OK] Generated {len(index)} slow-audio files (kokoro).")
    print(f"Index saved to: {index_path}")


if __name__ == "__main__":
    main()



