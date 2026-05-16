"""Streamlit app for the agentic video generator workspace.

Run from the repo root or scripts folder with:
    streamlit run scripts/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import html
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import requests
import streamlit as st
from config_helper import generate_aesthetic_style, generate_guidelines, generate_narration_style
from deepseek_utils import DEFAULT_DEEPSEEK_MODEL, DEEPSEEK_MODEL_CHOICES
from dotenv import load_dotenv
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENV_PATH = REPO_ROOT / ".env"

OUTPUT_FOLDERS = [
    "output_jsons",
    "images",
    "rejected_images",
    "audios",
    "clips",
    "videos",
]

OUTPUT_PAGE_FOLDERS = [
    ("outline, script and image prompts", "output_jsons"),
    ("images", "images"),
    ("audios", "audios"),
    ("clips", "clips"),
    ("videos", "videos"),
]

SCRIPT_STAGES = [
    ("Generate Outline", "generate_sections.py", "outline_texts.json"),
    ("Revise Outline", "validate_outline.py", "outline_texts.json"),
    ("Generate Narration", "generate_script.py", "narration.json"),
    ("Revise Narration", "validate_narration.py", "narration.json"),
]

IMAGE_PROMPT_STAGE = ("Generate Image Prompts", "generate_image_prompts.py", "image_prompts.json")

WORKFLOW_PAGES = [
    "Config Wizard",
    "Script Generation",
    "Image Generation",
    "Voice Generation",
    "Video Generation",
]

WORKFLOW_PILL_LABELS = {
    "Config Wizard": "Generate Config",
    "Script Generation": "Generate Scripts",
    "Image Generation": "Generate Images",
    "Voice Generation": "Generate Voices",
    "Video Generation": "Compile Video",
}

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: #101c2f;
            --surface-2: #16263f;
            --ink: #f3fbff;
            --muted: #afd0e8;
            --line: #2e496d;
            --accent: #1d4ed8;
            --accent-2: #15803d;
            --accent-3: #ffd166;
            --accent-soft: rgba(32, 227, 178, .16);
            --warn-soft: rgba(255, 209, 102, .18);
            --page: #07111f;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(76, 201, 240, .22), transparent 34%),
                linear-gradient(180deg, #07111f 0%, #0b1628 48%, #08111f 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 4rem;
            max-width: 1180px;
            font-size: 1.247rem;
        }

        .stMarkdown, .stText, .stCaption, p, label, div[data-testid="stMarkdownContainer"] {
            color: var(--ink);
            font-size: 1.197rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        h2, h3 {
            font-size: 1.917rem;
        }

        .app-hero {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            background:
                linear-gradient(135deg, rgba(32, 227, 178, .18) 0%, rgba(76, 201, 240, .12) 58%, rgba(255, 209, 102, .08) 100%);
            margin-bottom: 12px;
            box-shadow: 0 10px 26px rgba(0, 0, 0, .22);
        }

        .app-hero h1 {
            margin: 0;
            font-size: 1.82rem;
            line-height: 1.05;
        }

        .workflow-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            margin-top: 10px;
            align-items: center;
        }

        .workflow-pill {
            display: inline-flex;
            align-items: center;
            min-height: 1.55rem;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid #315173;
            background: rgba(15, 31, 53, .94);
            color: #cde3f7;
            font-size: .92rem;
            font-weight: 900;
            line-height: 1;
            white-space: nowrap;
        }

        .workflow-pill.active {
            background: #39ff14;
            border-color: rgba(198, 255, 190, .9);
            color: #04120a;
            box-shadow: 0 0 18px rgba(57, 255, 20, .42);
        }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            background: var(--surface);
            box-shadow: 0 12px 26px rgba(0, 0, 0, .22);
        }

        .metric-label {
            color: var(--muted);
            font-size: 1.067rem;
            text-transform: uppercase;
            letter-spacing: .04em;
        }

        .metric-value {
            color: var(--ink);
            font-weight: 700;
            font-size: 1.447rem;
            margin-top: 4px;
        }

        .path-box {
            background: #07111f;
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 10px 12px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 1.117rem;
            color: #d8f3ff;
            word-break: break-all;
        }

        .wizard-step-title {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 850;
            line-height: 1.25;
            margin: 10px 0 18px 0;
        }

        .stage-row {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px 18px;
            background: var(--surface);
            margin-bottom: 10px;
            box-shadow: 0 12px 26px rgba(0, 0, 0, .20);
            color: var(--ink);
            font-size: 1.217rem;
        }

        div.stButton > button[kind="primary"],
        div.stFormSubmitButton > button,
        div.stButton > button,
        div.stDownloadButton > button {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, .12);
            font-size: 1.227rem;
            font-weight: 850;
            min-height: 2.8rem;
        }

        div.stButton > button:disabled {
            background: #263a57;
            color: #88a5bd;
            border-color: #385372;
        }

        div.stButton > button[kind="primary"] {
            background: #8b0000;
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, .24);
            font-size: 1.48rem;
            font-weight: 950;
            min-height: 4.25rem;
            box-shadow: 0 18px 42px rgba(139, 0, 0, .30);
        }

        div.stButton > button[kind="primary"]:disabled {
            background: #3a3048;
            color: #a99bb7;
            border-color: #544664;
            box-shadow: none;
        }

        details[data-testid="stExpander"] > summary {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            border: 1px solid rgba(255, 255, 255, .12);
            border-radius: 8px;
            color: #ffffff !important;
            font-size: 1.227rem;
            font-weight: 850;
            min-height: 2.8rem;
            padding: .55rem .85rem;
        }

        details[data-testid="stExpander"] > summary p,
        details[data-testid="stExpander"] > summary span {
            color: #ffffff !important;
            font-weight: 850;
        }

        details[data-testid="stExpander"] {
            border-color: rgba(255, 255, 255, .12) !important;
            background: rgba(16, 28, 47, .65);
            border-radius: 8px;
        }

        section[data-testid="stSidebar"] {
            background: #091426;
            border-right: 1px solid var(--line);
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            font-size: 1rem;
        }

        div[data-baseweb="input"],
        div[data-baseweb="textarea"],
        div[data-baseweb="select"] {
            font-size: 1.187rem;
        }

        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stDecoration"] {
            background: #07111f !important;
            color: var(--ink) !important;
        }

        header[data-testid="stHeader"] *,
        div[data-testid="stToolbar"] *,
        div[data-testid="stStatusWidget"] * {
            background-color: #07111f !important;
            color: var(--ink) !important;
        }

        .stage-pill {
            display: inline-block;
            width: 58px;
            flex: 0 0 auto;
            margin: 0;
            padding: 3px 6px;
            border-radius: 999px;
            text-align: center;
            font-size: .68rem;
            font-weight: 900;
            letter-spacing: .05em;
            line-height: 1;
        }

        .stage-pill-done {
            background: #16a34a;
            color: #f0fff4;
            border: 1px solid rgba(134, 239, 172, .45);
        }

        .stage-pill-todo {
            background: #334155;
            color: #cbd5e1;
            border: 1px solid #475569;
        }

        .stage-pill-files {
            background: #1e3a8a;
            color: #dbeafe;
            border: 1px solid rgba(147, 197, 253, .45);
        }

        .stage-nav-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            min-height: 1.6rem;
            margin: 0;
            padding: 2px 4px;
            border: 1px solid transparent;
            border-radius: 6px;
            color: var(--ink) !important;
            text-decoration: none !important;
        }

        .stage-nav-row:hover {
            background: rgba(255, 255, 255, .08);
            border-color: rgba(255, 255, 255, .14);
        }

        .stage-nav-row.active {
            background: rgba(29, 78, 216, .24);
            border-color: rgba(134, 239, 172, .35);
        }

        .stage-nav-label {
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 1rem;
            font-weight: 850;
            line-height: 1.05;
        }

        section[data-testid="stSidebar"] div.stButton > button {
            background: transparent;
            color: var(--ink);
            border: 1px solid transparent;
            justify-content: flex-start;
            font-size: 1rem;
            font-weight: 850;
            min-height: 1.55rem;
            padding-left: .3rem;
            padding-top: .1rem;
            padding-bottom: .1rem;
            margin: 0;
            line-height: 1.05;
        }

        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
            gap: .18rem;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stElementContainer"] {
            margin-bottom: 0 !important;
        }

        section[data-testid="stSidebar"] div.stButton {
            margin: 0 !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
            margin-bottom: 0;
        }

        section[data-testid="stSidebar"] div.stButton > button:hover {
            background: rgba(255, 255, 255, .08);
            border-color: rgba(255, 255, 255, .14);
        }

        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
            background: rgba(29, 78, 216, .24);
            border-color: rgba(134, 239, 172, .35);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def sanitize_project_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", value.strip().replace(" ", "_"))
    if not cleaned:
        raise ValueError("Project name must contain at least one letter or number.")
    return cleaned


def project_root(project: str) -> Path:
    return REPO_ROOT / project


def source_dir(project: str) -> Path:
    return project_root(project) / "source_material"


def outputs_dir(project: str) -> Path:
    return project_root(project) / "outputs"


def output_dir(project: str, name: str) -> Path:
    return outputs_dir(project) / name


def ensure_project_dirs(project: str) -> None:
    source_dir(project).mkdir(parents=True, exist_ok=True)
    for folder in OUTPUT_FOLDERS:
        output_dir(project, folder).mkdir(parents=True, exist_ok=True)


def list_projects() -> list[str]:
    projects: list[str] = []
    for item in sorted(REPO_ROOT.iterdir()):
        if item.is_dir() and (item / "source_material").exists():
            projects.append(item.name)
    return projects


def default_config(project: str) -> dict:
    template_path = REPO_ROOT / "config.template.json"
    if template_path.exists():
        with template_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        config.setdefault("_project_config", {})["project_name"] = project
        return config

    return {
        "video_title": "",
        "n_section": 6,
        "section_outlines": [
            "Opening and setup",
            "Historical or narrative context",
            "Core story development",
            "Turning point",
            "Resolution or analysis",
            "Conclusion",
        ],
        "reference_links": [],
        "narration_style": [
            "Use a clear, serious, engaging tone.",
            "Maintain smooth transitions between sections.",
            "Avoid cliches and shallow language.",
        ],
        "intro_material": [],
        "source_material": [],
        "historical_context": "",
        "characters": {},
        "aesthetic_style": "cinematic historical realism, natural light, detailed composition",
        "_project_config": {
            "project_name": project,
            "image_config": {"model": "seedream-v4"},
            "narration_config": {"words_per_section": 400, "frames_per_section": 2},
            "validator_config": {"model": DEFAULT_DEEPSEEK_MODEL},
            "tts_config": {"model": "kokoro", "voice_id": "af"},
        },
    }


def ensure_config_defaults(config: dict, project: str) -> dict:
    project_config = config.setdefault("_project_config", {})
    project_config.setdefault("project_name", project)
    if not isinstance(project_config.get("image_config"), dict):
        project_config["image_config"] = {"model": "seedream-v4"}
    if not isinstance(project_config.get("narration_config"), dict):
        project_config["narration_config"] = {"words_per_section": 400, "frames_per_section": 2}
    if not isinstance(project_config.get("validator_config"), dict):
        project_config["validator_config"] = {"model": DEFAULT_DEEPSEEK_MODEL}
    if not isinstance(project_config.get("tts_config"), dict):
        project_config["tts_config"] = {"model": "kokoro", "voice_id": "af"}
    project_config["image_config"].setdefault("model", "seedream-v4")
    project_config["narration_config"].setdefault("words_per_section", 400)
    project_config["narration_config"].setdefault("frames_per_section", 2)
    project_config["validator_config"].setdefault("model", DEFAULT_DEEPSEEK_MODEL)
    project_config["tts_config"].setdefault("model", "kokoro")
    project_config["tts_config"].setdefault("voice_id", "af")
    return config


def config_path(project: str) -> Path:
    return source_dir(project) / "config.json"


def app_state_path(project: str) -> Path:
    return source_dir(project) / "app_state.json"


def load_config(project: str) -> dict:
    path = config_path(project)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return ensure_config_defaults(data, project)
    return ensure_config_defaults(default_config(project), project)


def save_config(project: str, config: dict) -> Path:
    ensure_project_dirs(project)
    config = ensure_config_defaults(config, project)
    config["_project_config"]["project_name"] = project
    path = config_path(project)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return path


def load_app_state(project: str) -> dict:
    path = app_state_path(project)
    if not path.exists():
        return {"completed_stages": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"completed_stages": {}}
    if not isinstance(data, dict):
        return {"completed_stages": {}}
    completed = data.get("completed_stages", {})
    if not isinstance(completed, dict):
        completed = {}
    data["completed_stages"] = completed
    return data


def save_app_state(project: str, state: dict) -> Path:
    ensure_project_dirs(project)
    path = app_state_path(project)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return path


def set_stage_complete(project: str, stage: str, complete: bool = True) -> None:
    state = load_app_state(project)
    completed = state.setdefault("completed_stages", {})
    completed[stage] = {
        "complete": bool(complete),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_app_state(project, state)


def stage_acknowledged(project: str, stage: str) -> bool:
    state = load_app_state(project)
    entry = state.get("completed_stages", {}).get(stage, {})
    return bool(isinstance(entry, dict) and entry.get("complete"))


def split_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def parse_characters(value: str) -> dict:
    characters: dict[str, str] = {}
    for line in split_lines(value):
        if ":" in line:
            name, desc = line.split(":", 1)
            name = name.strip()
            desc = desc.strip()
            if name and desc:
                characters[name] = desc
    return characters


def characters_to_text(characters: dict) -> str:
    return "\n".join(f"{name}: {desc}" for name, desc in characters.items())


def load_image_model_map() -> dict[str, str]:
    path = SCRIPT_DIR / "image_models.txt"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("image_models.txt must contain a JSON dictionary.")
    return {str(label).strip(): str(model_id).strip() for label, model_id in data.items() if str(label).strip() and str(model_id).strip()}


def image_model_label_for_config(value: str) -> str:
    model_map = load_image_model_map()
    value = (value or "seedream-v4").strip()
    if value in model_map:
        return value
    for label, model_id in model_map.items():
        if value == model_id:
            return label
    return value


def load_voice_map() -> dict[str, bool]:
    path = SCRIPT_DIR / "voices.txt"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("voices.txt must contain a JSON dictionary.")
    return {str(voice).strip(): bool(is_inworld) for voice, is_inworld in data.items() if str(voice).strip()}


def normalized_tts_model(value: str | None) -> str:
    model = (value or "kokoro").strip().lower()
    if model == "inworld":
        return "inworld-tts-1.5-max"
    if model in {"kokoro", "inworld-tts-1.5-max", "inworld-tts-2"}:
        return model
    return "kokoro"


def voice_options_for_tts_model(tts_model: str) -> list[str]:
    use_inworld = normalized_tts_model(tts_model).startswith("inworld-tts-")
    voices = [voice for voice, is_inworld in load_voice_map().items() if is_inworld == use_inworld]
    fallback = ["Ashley"] if use_inworld else ["af"]
    return voices or fallback


def selectbox_options_with_current(options: list[str], current: str, fallback: str) -> tuple[list[str], str]:
    normalized = list(dict.fromkeys(options))
    selected = current or fallback
    if selected not in normalized:
        normalized.append(selected)
    return normalized, selected


def safe_filename_from_url(url: str, fallback_ext: str = "") -> str:
    parsed = urlparse(url)
    name = unquote(Path(parsed.path).name).strip()
    name = re.sub(r"[^A-Za-z0-9_. -]+", "_", name)
    if not name:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        name = f"source_{digest}{fallback_ext}"
    if fallback_ext and not name.lower().endswith(fallback_ext):
        name = f"{name}{fallback_ext}"
    return name


def classify_url(url: str) -> str:
    if "wikipedia.org/wiki/" in url.lower():
        return "reference"

    lower_path = urlparse(url).path.lower()
    if lower_path.endswith(".pdf"):
        return "pdf"
    if lower_path.endswith(".txt"):
        return "txt"

    try:
        response = requests.head(url, allow_redirects=True, timeout=12)
        content_type = response.headers.get("content-type", "").lower()
    except requests.RequestException:
        content_type = ""

    if "application/pdf" in content_type:
        return "pdf"
    if "text/plain" in content_type:
        return "txt"
    return "reference"


def download_source_link(url: str, project: str) -> str:
    kind = classify_url(url)
    if kind not in {"pdf", "txt"}:
        return ""

    ext = ".pdf" if kind == "pdf" else ".txt"
    filename = safe_filename_from_url(url, ext)
    target = source_dir(project) / filename

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)
    return filename


def save_uploaded_sources(project: str, files) -> list[str]:
    saved: list[str] = []
    for uploaded in files or []:
        filename = re.sub(r"[^A-Za-z0-9_. -]+", "_", uploaded.name)
        target = source_dir(project) / filename
        target.write_bytes(uploaded.getbuffer())
        saved.append(filename)
    return saved


def output_status(project: str) -> dict[str, bool]:
    base = output_dir(project, "output_jsons")
    video_path = output_dir(project, "videos") / f"{project}.mp4"
    return {
        "outline_texts.json": (base / "outline_texts.json").exists(),
        "narration.json": (base / "narration.json").exists(),
        "image_prompts.json": (base / "image_prompts.json").exists(),
        "tts_index.json": (base / "tts_index.json").exists(),
        "video": video_path.exists() and video_path.stat().st_size > 0,
    }


def config_warnings(config: dict) -> list[str]:
    warnings: list[str] = []
    if not config.get("video_title", "").strip() or config.get("video_title") == "Your Video Title Here":
        warnings.append("Video title is empty or still a template value.")
    section_outlines = config.get("section_outlines", [])
    if not section_outlines:
        warnings.append("No guidelines are saved.")
    elif any("Section 1: Opening and introduction" in str(item) for item in section_outlines):
        warnings.append("Guidelines still look like template defaults.")
    if not config.get("narration_style", []):
        warnings.append("No narration style is saved.")
    if not config.get("aesthetic_style", "").strip():
        warnings.append("Aesthetic style is empty.")
    if not config.get("_project_config", {}).get("project_name"):
        warnings.append("Project name is missing from _project_config.")
    return warnings


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"

    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def progress_event_from_line(line: str) -> tuple[str, str, int, int, bool] | None:
    patterns = [
        (
            r"Generating narration section\s+(\d+)\s*/\s*(\d+)",
            "narration",
            "Generating narration section {current} of {total}",
            False,
        ),
        (
            r"Generating image prompt\s+(\d+)\s*/\s*(\d+)",
            "image_prompts",
            "Generating image prompt {current} of {total}",
            False,
        ),
        (
            r"Generating audio\s+(\d+)\s*/\s*(\d+)",
            "audio",
            "Generating audio {current} of {total}",
            False,
        ),
        (
            r"Generating clip\s+(\d+)\s*/\s*(\d+)",
            "clips",
            "Generating clip {current} of {total}",
            False,
        ),
        (
            r"Loading final video clip\s+(\d+)\s*/\s*(\d+)",
            "video_load",
            "Loading final video clip {current} of {total}",
            False,
        ),
        (
            r"Writing final video\s+(\d+)\s*/\s*(\d+)",
            "video_write",
            "Writing final video {current} of {total}",
            True,
        ),
    ]
    for pattern, key, template, current_is_complete in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            current = int(match.group(1))
            total = max(1, int(match.group(2)))
            return template.format(current=current, total=total), key, current, total, current_is_complete
    return None


def run_script(script_name: str, project: str, extra_args: list[str] | None = None) -> tuple[int, str]:
    script_path = SCRIPT_DIR / script_name
    command = [sys.executable, str(script_path), project]
    if extra_args:
        command.extend(extra_args)
    process = subprocess.Popen(
        command,
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    lines: list[str] = []
    progress_text = st.empty()
    progress_bar = st.empty()
    saw_progress = False
    progress_key = None
    progress_started_at = 0.0

    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        lines.append(line)
        progress_event = progress_event_from_line(line)
        if progress_event:
            message, key, current, total, current_is_complete = progress_event
            saw_progress = True
            if key != progress_key:
                progress_key = key
                progress_started_at = time.monotonic()

            completed = current if current_is_complete else max(0, current - 1)
            if key in {"video_load", "video_write"}:
                eta_text = "estimating time remaining"
                elapsed = time.monotonic() - progress_started_at
                if 0 < completed < total and elapsed >= 1:
                    seconds_per_unit = elapsed / completed
                    eta_text = f"about {format_duration(seconds_per_unit * (total - completed))} remaining"
                elif completed >= total:
                    eta_text = "almost done"

                progress_text.info(f"{message} - {eta_text}")
            else:
                progress_text.info(message)
            progress_bar.progress(min(1.0, current / total))

    return_code = process.wait()
    if saw_progress:
        progress_bar.progress(1.0)
        time.sleep(0.25)
    progress_text.empty()
    progress_bar.empty()
    full_log = "\n".join(lines)
    return return_code, full_log


def artifact_location(project: str, expected: str | None) -> Path | None:
    if not expected:
        return None
    if expected in {"outline_texts.json", "narration.json"}:
        return output_dir(project, "output_jsons") / expected
    if expected == "clips":
        return output_dir(project, "clips")
    if expected == "videos":
        return final_video_path(project)
    return None


def final_video_path(project: str) -> Path:
    return output_dir(project, "videos") / f"{project}.mp4"


def archive_existing_final_video(project: str) -> Path | None:
    path = final_video_path(project)
    if not path.exists():
        return None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}_previous_{timestamp}{path.suffix}")
    counter = 2
    while backup_path.exists():
        backup_path = path.with_name(f"{path.stem}_previous_{timestamp}_{counter}{path.suffix}")
        counter += 1

    path.replace(backup_path)
    return backup_path


def paths_match(left: str, right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except Exception:
        normalized_left = str(left).replace("\\", "/").lower().rstrip("/")
        normalized_right = str(right).replace("\\", "/").lower().rstrip("/")
        return normalized_left == normalized_right


def stage_success_text(label: str, expected: str | None) -> str:
    if expected == "outline_texts.json":
        return "Outline generated." if "Generate" in label else "Outline revised."
    if expected == "narration.json":
        return "Narration generated." if "Generate" in label else "Narration revised."
    if expected == "image_prompts.json":
        return "Image prompts generated."
    if expected == "tts_index.json":
        return "Audio clips generated."
    if expected == "clips":
        return "Video clips generated."
    if expected == "videos":
        return "Final video generated."
    return f"{label} completed."


def media_key_from_name(path: Path, prefix: str) -> tuple[int, int] | None:
    match = re.match(rf"^{re.escape(prefix)}_(\d+)_(\d+)$", path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def media_keys_in_dir(path: Path, prefix: str, extensions: tuple[str, ...]) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    if not path.exists():
        return keys
    for item in path.iterdir():
        if not item.is_file() or item.suffix.lower() not in extensions:
            continue
        key = media_key_from_name(item, prefix)
        if key is not None:
            keys.add(key)
    return keys


def format_media_key(key: tuple[int, int]) -> str:
    return f"{key[0]}_{key[1]}"


def narration_segment_keys(project: str) -> set[tuple[int, int]]:
    path = output_dir(project, "output_jsons") / "narration.json"
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()

    keys: set[tuple[int, int]] = set()
    for section_index, section in enumerate(data.get("sections", []), start=1):
        for prompt_index, text in enumerate(section.get("narration_text", []), start=1):
            if str(text).strip():
                keys.add((section_index, prompt_index))
    return keys


def image_prompt_keys(project: str) -> set[tuple[int, int]]:
    path = output_dir(project, "output_jsons") / "image_prompts.json"
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()

    keys: set[tuple[int, int]] = set()
    for section_index, section in enumerate(data.get("sections", []), start=1):
        for prompt_index, prompt in enumerate(section.get("image_prompts", []), start=1):
            if str(prompt).strip():
                keys.add((section_index, prompt_index))
    return keys


def parse_media_key_text(value: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)[_\-:](\d+)\s*$", value)
    if not match:
        raise ValueError(f"Invalid frame key: {value}. Use section_segment, e.g. 3_2.")
    return int(match.group(1)), int(match.group(2))


def parse_media_selection_text(selection: str, available_keys: set[tuple[int, int]]) -> set[tuple[int, int]]:
    selection = selection.strip().lower()
    available = sorted(available_keys)
    if selection == "all":
        return set(available)
    if "-" in selection and "," not in selection:
        start_text, end_text = selection.split("-", 1)
        start_key = parse_media_key_text(start_text)
        end_key = parse_media_key_text(end_text)
        if start_key > end_key:
            start_key, end_key = end_key, start_key
        return {key for key in available if start_key <= key <= end_key}
    requested = {parse_media_key_text(part) for part in selection.split(",") if part.strip()}
    missing = requested - set(available)
    if missing:
        missing_text = ", ".join(format_media_key(key) for key in sorted(missing))
        raise ValueError(f"Requested frames are not available: {missing_text}")
    return requested


def stage_readiness(project: str, stage: str) -> tuple[bool, str]:
    status = output_status(project)

    if stage == "Config Wizard":
        warnings = config_warnings(load_config(project))
        if warnings:
            return False, "Config still has required fields missing."
        return True, "Config is ready."

    if stage == "Script Generation":
        if not status["outline_texts.json"]:
            return False, "outline_texts.json has not been generated."
        if not status["narration.json"]:
            return False, "narration.json has not been generated."
        return True, "Outline and narration JSON files are ready."

    if stage == "Image Generation":
        prompt_keys = image_prompt_keys(project)
        if not prompt_keys:
            return False, "image_prompts.json has not been generated."
        approved_image_keys = media_keys_in_dir(output_dir(project, "images"), "image", (".png",))
        missing = sorted(prompt_keys - approved_image_keys)
        if missing:
            missing_text = ", ".join(format_media_key(key) for key in missing[:8])
            if len(missing) > 8:
                missing_text += ", ..."
            return False, f"Approved images are missing for: {missing_text}"
        return True, "Image prompts and approved images are ready."

    if stage == "Voice Generation":
        narration_keys = narration_segment_keys(project)
        if not narration_keys:
            return False, "narration.json has no narration frames."
        audio_keys = media_keys_in_dir(output_dir(project, "audios"), "audio", (".mp3", ".wav"))
        missing = sorted(narration_keys - audio_keys)
        if missing:
            missing_text = ", ".join(format_media_key(key) for key in missing[:8])
            if len(missing) > 8:
                missing_text += ", ..."
            return False, f"Audio files are missing for: {missing_text}"
        return True, "Audio files are ready."

    if stage == "Video Generation":
        if not status["video"]:
            return False, "Final video MP4 has not been generated."
        return True, "Final video is ready."

    return False, "Unknown stage."


def stage_completion_status(project: str) -> dict[str, bool]:
    return {stage: stage_acknowledged(project, stage) for stage in WORKFLOW_PAGES}


def render_sidebar_navigation(project: str, pages: list[str]) -> str:
    completion = stage_completion_status(project)
    st.sidebar.markdown("### Stage Status")

    for page_name in pages:
        if page_name == "Outputs":
            pill_text = "FILES"
            pill_class = "stage-pill-files"
        elif completion.get(page_name, False):
            pill_text = "DONE"
            pill_class = "stage-pill-done"
        else:
            pill_text = "TODO"
            pill_class = "stage-pill-todo"

        active_class = " active" if st.session_state.workspace_page == page_name else ""
        href = f"?stage={quote(page_name)}"
        st.sidebar.markdown(
            f"""
            <a class="stage-nav-row{active_class}" href="{href}" target="_self">
                <span class="stage-nav-label">{html.escape(page_name)}</span>
                <span class="stage-pill {pill_class}">{pill_text}</span>
            </a>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.workspace_page


def selected_stage_from_query() -> str:
    try:
        value = st.query_params.get("stage", "")
    except Exception:
        try:
            params = st.experimental_get_query_params()
            value = params.get("stage", [""])
        except Exception:
            value = ""

    if isinstance(value, list):
        value = value[0] if value else ""
    return unquote(str(value))


def verify_stage_artifact(
    project: str,
    expected: str | None,
    expected_clip_keys: set[tuple[int, int]] | None = None,
    expected_audio_keys: set[tuple[int, int]] | None = None,
    min_mtime: float | None = None,
    log_text: str | None = None,
) -> tuple[bool, str]:
    if not expected:
        return True, ""

    if expected.endswith(".json"):
        path = output_dir(project, "output_jsons") / expected
        if not path.exists():
            return False, f"Expected output was not created: {path}"
        if path.stat().st_size <= 2:
            return False, f"Expected output is empty: {path}"
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            return False, f"Expected output is not valid JSON: {path} ({exc})"

        if expected == "outline_texts.json":
            if not isinstance(data, list) or not data:
                return False, f"outline_texts.json contains no sections: {path}"
            for index, item in enumerate(data, start=1):
                if not isinstance(item, dict):
                    return False, f"outline_texts.json item {index} is not an object."
                if not item.get("section_title") or not item.get("outline"):
                    return False, f"outline_texts.json item {index} is missing section_title or outline."

        if expected == "narration.json":
            sections = data.get("sections", []) if isinstance(data, dict) else []
            if not sections:
                return False, f"narration.json contains no sections: {path}"

        if expected == "image_prompts.json":
            sections = data.get("sections", []) if isinstance(data, dict) else []
            if not sections or not any(section.get("image_prompts") for section in sections):
                return False, f"image_prompts.json contains no image prompts: {path}"

        if expected == "tts_index.json":
            if not isinstance(data, list) or not data:
                return False, f"tts_index.json contains no audio entries: {path}"
            audio_keys = media_keys_in_dir(output_dir(project, "audios"), "audio", (".mp3", ".wav"))
            keys_to_check = expected_audio_keys if expected_audio_keys is not None else narration_segment_keys(project)
            if keys_to_check:
                missing = sorted(keys_to_check - audio_keys)
                if missing:
                    missing_text = ", ".join(format_media_key(key) for key in missing)
                    return False, f"Missing audio files for frame keys: {missing_text}"

        return True, f"Verified output: {path}"

    if expected == "clips":
        path = output_dir(project, "clips")
        image_keys = media_keys_in_dir(output_dir(project, "images"), "image", (".png",))
        clip_keys = media_keys_in_dir(path, "image", (".mp4",))
        keys_to_check = expected_clip_keys if expected_clip_keys is not None else image_keys
        if not image_keys:
            return False, f"No approved image files found in: {output_dir(project, 'images')}"
        if not keys_to_check:
            return True, f"No selected clips needed generation in: {path}"
        if not clip_keys:
            return False, f"No clip MP4 files found in: {path}"
        missing = sorted(keys_to_check - clip_keys)
        if missing:
            missing_text = ", ".join(f"{section}_{segment}" for section, segment in missing)
            return False, f"Missing clip MP4 files for image keys: {missing_text}"
        return True, f"Verified {len(keys_to_check)} selected clips in: {path}"

    if expected == "videos":
        video_path = final_video_path(project)
        log_text = log_text or ""
        success_match = re.search(r"COMPOSE_FINAL_VIDEO_OK:\s*(.+)", log_text)
        if not success_match:
            return False, "make_video.py did not report COMPOSE_FINAL_VIDEO_OK for this run."
        success_path = success_match.group(1).strip()
        if not paths_match(success_path, video_path):
            return False, f"make_video.py reported a different final video path: {success_path}"

        size_match = re.search(r"COMPOSE_FINAL_VIDEO_SIZE:\s*(\d+)", log_text)
        if not size_match:
            return False, "make_video.py did not report COMPOSE_FINAL_VIDEO_SIZE for this run."
        reported_size = int(size_match.group(1))

        if not video_path.exists():
            return False, f"Expected final video was not created: {video_path}"
        actual_size = video_path.stat().st_size
        if actual_size <= 0:
            return False, f"Expected final video is empty: {video_path}"
        if actual_size != reported_size:
            return False, f"Final video size mismatch: script reported {reported_size}, file is {actual_size}."
        if min_mtime is not None and video_path.stat().st_mtime < min_mtime:
            modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(video_path.stat().st_mtime))
            return False, f"Final video was not updated by this run: {video_path} (last modified {modified})"
        return True, f"Verified video: {video_path}"

    return True, ""


def tts_script_for_config(config: dict) -> str:
    model = normalized_tts_model(
        config.get("_project_config", {})
        .get("tts_config", {})
        .get("model", "kokoro")
    )
    if model.startswith("inworld-tts-"):
        return "generate_inworld_voice.py"
    return "generate_kokoro_voice.py"


def run_stage(
    project: str,
    label: str,
    script_name: str,
    expected_output: str | None = None,
    extra_args: list[str] | None = None,
    expected_clip_keys: set[tuple[int, int]] | None = None,
    expected_audio_keys: set[tuple[int, int]] | None = None,
) -> bool:
    started_at = time.time()
    with st.spinner(f"Running {label}... (DO NOT INTERRUPT)"):
        code, log = run_script(script_name, project, extra_args=extra_args)
    if code == 0:
        ok, message = verify_stage_artifact(
            project,
            expected_output,
            expected_clip_keys=expected_clip_keys,
            expected_audio_keys=expected_audio_keys,
            min_mtime=started_at,
            log_text=log,
        )
        if ok:
            location = artifact_location(project, expected_output)
            st.success(stage_success_text(label, expected_output))
            if location:
                st.markdown(f"<div class='path-box'>{location}</div>", unsafe_allow_html=True)
            return True
        else:
            st.error(f"{label} finished but output validation failed.")
            st.write(message)
            recent_log = "\n".join(log.splitlines()[-40:])
            with st.expander("Recent log output", expanded=False):
                st.code(recent_log or "No log output captured.", language="text")
            return False
    else:
        st.error(f"{label} failed with exit code {code}.")
        recent_log = "\n".join(log.splitlines()[-40:])
        with st.expander("Recent log output", expanded=False):
            st.code(recent_log or "No log output captured.", language="text")
        return False


def compose_final_video(project: str) -> bool:
    video_path = final_video_path(project)
    archived_path = archive_existing_final_video(project)
    started_at = time.time()

    with st.spinner("Running Compose Final Video... (DO NOT INTERRUPT)"):
        code, log = run_script("make_video.py", project)

    if code != 0:
        st.error(f"Compose Final Video failed with exit code {code}.")
        if archived_path:
            st.info(f"Previous video was archived at: {archived_path}")
        recent_log = "\n".join(log.splitlines()[-40:])
        with st.expander("Recent log output", expanded=True):
            st.code(recent_log or "No log output captured.", language="text")
        return False

    ok, message = verify_stage_artifact(
        project,
        "videos",
        min_mtime=started_at,
        log_text=log,
    )
    if not ok:
        st.error("Compose Final Video finished but output validation failed.")
        st.write(message)
        if archived_path:
            st.info(f"Previous video was archived at: {archived_path}")
        recent_log = "\n".join(log.splitlines()[-40:])
        with st.expander("Recent log output", expanded=True):
            st.code(recent_log or "No log output captured.", language="text")
        return False

    st.success("Final video generated.")
    st.markdown(f"<div class='path-box'>{video_path}</div>", unsafe_allow_html=True)
    if archived_path:
        st.caption(f"Previous video archived as: {archived_path.name}")
    return True


def run_stage_quiet(
    project: str,
    label: str,
    script_name: str,
    expected_output: str | None = None,
) -> tuple[bool, str]:
    with st.spinner(f"Running {label}...(DO NOT INTERRUPT)"):
        code, log = run_script(script_name, project)
    if code != 0:
        recent_log = "\n".join(log.splitlines()[-40:])
        return False, f"{label} failed with exit code {code}.\n{recent_log}".strip()

    ok, message = verify_stage_artifact(project, expected_output)
    if not ok:
        recent_log = "\n".join(log.splitlines()[-40:])
        detail = "\n".join(part for part in [message, recent_log] if part)
        return False, detail or f"{label} finished but output validation failed."
    return True, ""


def load_image_prompt_items(project: str) -> list[dict]:
    path = output_dir(project, "output_jsons") / "image_prompts.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items: list[dict] = []
    for section_index, section in enumerate(data.get("sections", []), start=1):
        prompts = section.get("image_prompts", [])
        narrations = section.get("narration_text", [])
        for prompt_index, prompt in enumerate(prompts, start=1):
            narration = narrations[prompt_index - 1] if prompt_index - 1 < len(narrations) else ""
            items.append(
                {
                    "section_index": section_index,
                    "prompt_index": prompt_index,
                    "section_title": section.get("section_title", f"Section {section_index}"),
                    "narration": narration,
                    "prompt": prompt,
                }
            )
    return items


def image_model_for_config(config: dict) -> str:
    model_value = (
        config.get("_project_config", {})
        .get("image_config", {})
        .get("model", "seedream-v4")
    )
    model_value = (model_value or "seedream-v4").strip()
    return load_image_model_map().get(model_value, model_value)


def generate_image_with_fal(prompt: str, config: dict) -> Image.Image:
    try:
        import fal_client
    except ImportError as exc:
        raise RuntimeError("Missing fal-client. Install dependencies from scripts/requirements.txt.") from exc

    load_dotenv(dotenv_path=ENV_PATH)
    fal_key = os.getenv("FAL_KEY")
    if not fal_key:
        raise RuntimeError("Missing FAL_KEY in the repo .env file.")
    os.environ["FAL_KEY"] = fal_key

    result = fal_client.subscribe(
        image_model_for_config(config),
        arguments={
            "prompt": prompt,
            "image_size": {"height": 1152, "width": 2048},
            "num_images": 1,
            "max_images": 1,
            "enable_safety_checker": False,
            "enhance_prompt_mode": "standard",
        },
        with_logs=False,
    )

    if "images" not in result or not result["images"]:
        raise RuntimeError("Image API returned no images.")

    image_url = result["images"][0]["url"]
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def save_image_checked(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Image was not saved correctly: {path}")


def copy_checked(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if not dst.exists() or dst.stat().st_size <= 0:
        raise RuntimeError(f"Image was not copied correctly: {dst}")


def init_state() -> None:
    st.session_state.setdefault("config_step", 0)
    st.session_state.setdefault("selected_project", "")
    st.session_state.setdefault("workspace_page", "Config Wizard")
    st.session_state.setdefault("pending_workspace_page", "")
    st.session_state.setdefault("config_saved_notice", "")
    st.session_state.setdefault("active_review_index", 0)
    st.session_state.setdefault("active_review_path", "")
    st.session_state.setdefault("active_prompt_text", "")
    st.session_state.setdefault("image_review_notice", "")
    st.session_state.setdefault("image_review_notice_kind", "success")
    st.session_state.setdefault("link_results", [])


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def set_active_review_index(index: int, item_count: int) -> None:
    st.session_state.active_review_index = max(0, min(item_count - 1, index))
    st.session_state.active_review_path = ""
    st.session_state.active_prompt_text = ""


def render_header(active_page: str = "") -> None:
    pills = []
    for page_name, label in WORKFLOW_PILL_LABELS.items():
        active_class = " active" if page_name == active_page else ""
        pills.append(f'<span class="workflow-pill{active_class}">{html.escape(label)}</span>')

    st.markdown(
        f"""
        <div class="app-hero">
            <h1>Agentic Video Generator</h1>
            <div class="workflow-pills">{"".join(pills)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_selector() -> str:
    st.sidebar.header("Project")
    projects = list_projects()

    mode = st.sidebar.radio("Mode", ["Open existing", "Create new"], horizontal=False)
    if mode == "Create new":
        raw_name = st.sidebar.text_input("New project name", value=st.session_state.selected_project or "NewProject")
        try:
            project = sanitize_project_name(raw_name)
            if st.sidebar.button("Create / open project", width="stretch"):
                ensure_project_dirs(project)
                if not config_path(project).exists():
                    save_config(project, default_config(project))
                st.session_state.selected_project = project
                rerun_app()
        except ValueError as exc:
            st.sidebar.error(str(exc))
            project = ""
    else:
        if not projects:
            st.sidebar.info("No projects found yet. Create one to begin.")
            project = ""
        else:
            default_index = 0
            if st.session_state.selected_project in projects:
                default_index = projects.index(st.session_state.selected_project)
            project = st.sidebar.selectbox("Existing project", projects, index=default_index)
            if project:
                st.session_state.selected_project = project

    if not st.session_state.selected_project and project:
        st.session_state.selected_project = project
    return st.session_state.selected_project


def wizard_form_nav(next_label: str, previous_disabled: bool = False) -> tuple[bool, bool]:
    left, right = st.columns([1, 2])
    with left:
        previous = st.form_submit_button("Previous", disabled=previous_disabled, width="stretch")
    with right:
        submitted = st.form_submit_button(next_label, width="stretch")
    return previous, submitted


def wizard_button_nav(next_label: str, previous_disabled: bool = False) -> tuple[bool, bool]:
    left, right = st.columns([1, 2])
    with left:
        previous = st.button("Previous", disabled=previous_disabled, width="stretch")
    with right:
        submitted = st.button(next_label, width="stretch")
    return previous, submitted


def load_content_state_from_config(project: str, config: dict | None = None) -> None:
    config = config or load_config(project)
    narration_config = config.get("_project_config", {}).get("narration_config", {})

    st.session_state.content_state_project = project
    st.session_state.content_video_title = config.get("video_title", "")
    st.session_state.content_n_section = int(config.get("n_section", 6))
    st.session_state.content_words_per_section = int(narration_config.get("words_per_section", 400))
    st.session_state.content_frames_per_section = int(narration_config.get("frames_per_section", 2))
    st.session_state.content_guidelines = "\n".join(config.get("section_outlines", []))
    st.session_state.content_narration_style = "\n".join(config.get("narration_style", []))
    st.session_state.content_aesthetic_style = config.get("aesthetic_style", "")
    st.session_state.content_historical_context = config.get("historical_context", "")
    st.session_state.content_characters = characters_to_text(config.get("characters", {}))
    st.session_state.llm_help_guidelines = False
    st.session_state.llm_help_narration_style = False
    st.session_state.llm_help_aesthetic_style = False


def apply_content_basics(
    project: str,
    config: dict,
    title: str,
    n_section: int,
    words_per_section: int,
    frames_per_section: int,
) -> None:
    config["video_title"] = title.strip()
    config["n_section"] = int(n_section)
    config["_project_config"]["narration_config"] = {
        "words_per_section": int(words_per_section),
        "frames_per_section": int(frames_per_section),
    }
    config["_project_config"]["project_name"] = project


def save_content_basics(
    project: str,
    config: dict,
    title: str,
    n_section: int,
    words_per_section: int,
    frames_per_section: int,
) -> Path:
    apply_content_basics(project, config, title, n_section, words_per_section, frames_per_section)
    return save_config(project, config)


def save_content_page(
    project: str,
    config: dict,
    title: str,
    n_section: int,
    words_per_section: int,
    frames_per_section: int,
    guidelines: str,
    narration_style: str,
    aesthetic_style: str,
    historical_context: str,
    characters: str,
) -> Path:
    apply_content_basics(project, config, title, n_section, words_per_section, frames_per_section)
    config["section_outlines"] = split_lines(guidelines)
    config["narration_style"] = split_lines(narration_style)
    config["aesthetic_style"] = aesthetic_style.strip()
    config["historical_context"] = historical_context.strip()
    config["characters"] = parse_characters(characters)
    return save_config(project, config)


def save_models_page(
    project: str,
    config: dict,
    tts_model: str,
    voice_id: str,
    deepseek_model: str,
    image_model: str,
) -> Path:
    project_config = config.setdefault("_project_config", {})
    project_config["project_name"] = project
    project_config["tts_config"] = {"model": tts_model, "voice_id": voice_id.strip()}
    project_config["validator_config"] = {"model": deepseek_model}
    project_config["image_config"] = {"model": image_model.strip() or "seedream-v4"}
    return save_config(project, config)


def render_config_wizard(project: str) -> None:
    config = load_config(project)
    steps = ["Content", "Sources", "Models"]
    if st.session_state.config_step >= len(steps):
        st.session_state.config_step = len(steps) - 1

    st.subheader("Configuration Wizard")
    st.progress((st.session_state.config_step + 1) / len(steps))
    st.markdown(
        f"<div class='wizard-step-title'>Step {st.session_state.config_step + 1} of {len(steps)}: {steps[st.session_state.config_step]}</div>",
        unsafe_allow_html=True,
    )

    step = st.session_state.config_step

    if step == 0:
        if st.session_state.get("content_state_project") != project:
            load_content_state_from_config(project, config)

        title = st.text_input("Video topic", key="content_video_title")
        section_cols = st.columns(3)
        with section_cols[0]:
            n_section = st.number_input(
                "Number of sections",
                min_value=1,
                max_value=30,
                key="content_n_section",
            )
        with section_cols[1]:
            words_per_section = st.number_input(
                "Tentative words per section",
                min_value=100,
                max_value=3000,
                step=50,
                key="content_words_per_section",
            )
        with section_cols[2]:
            frames_per_section = st.number_input(
                "Tentative frames per section",
                min_value=1,
                max_value=12,
                step=1,
                key="content_frames_per_section",
            )

        if st.button("Save Config", key="save_content_basics", width="stretch"):
            save_content_basics(project, config, title, int(n_section), int(words_per_section), int(frames_per_section))
            st.success("Config saved.")

        if st.toggle("Use LLM help for Guidelines", key="llm_help_guidelines"):
            if st.button("Generate Guidelines", key="generate_guidelines_helper", width="stretch"):
                if not title.strip():
                    st.error("Enter a video topic before generating guidelines.")
                else:
                    with st.spinner("Generating guidelines..."):
                        st.session_state.content_guidelines = generate_guidelines(title.strip(), int(n_section))
                    st.success("Guidelines generated. Review and edit before saving.")
        updated = st.text_area("Guidelines", key="content_guidelines", height=220)

        if st.toggle("Use LLM help for Narration Style", key="llm_help_narration_style"):
            if st.button("Generate Narration Style", key="generate_narration_style_helper", width="stretch"):
                if not title.strip():
                    st.error("Enter a video topic before generating narration style.")
                else:
                    with st.spinner("Generating narration style..."):
                        st.session_state.content_narration_style = generate_narration_style(title.strip())
                    st.success("Narration style generated. Review and edit before saving.")
        narration_style = st.text_area("Narration Style", key="content_narration_style", height=150)

        if st.toggle("Use LLM help for Aesthetic Style", key="llm_help_aesthetic_style"):
            if st.button("Generate Aesthetic Style", key="generate_aesthetic_style_helper", width="stretch"):
                if not title.strip():
                    st.error("Enter a video topic before generating aesthetic style.")
                else:
                    with st.spinner("Generating aesthetic style..."):
                        st.session_state.content_aesthetic_style = generate_aesthetic_style(title.strip())
                    st.success("Aesthetic style generated. Review and edit before saving.")
        aesthetic_style = st.text_area("Aesthetic style", key="content_aesthetic_style", height=110)

        with st.expander("Advanced Options (Not mandatory)", expanded=False):
            historical_context = st.text_area("Historical context", key="content_historical_context", height=100)
            characters = st.text_area(
                "Character canon, one per line as Name: visual description",
                key="content_characters",
                height=170,
            )

        previous, submitted = wizard_button_nav("Save & continue to Sources", previous_disabled=True)
        if previous:
            st.session_state.config_step = max(0, st.session_state.config_step - 1)
            rerun_app()
        if submitted:
            save_content_page(
                project,
                config,
                title,
                int(n_section),
                int(words_per_section),
                int(frames_per_section),
                updated,
                narration_style,
                aesthetic_style,
                historical_context,
                characters,
            )
            st.success("Content saved.")
            st.session_state.config_step = min(len(steps) - 1, st.session_state.config_step + 1)
            rerun_app()

    elif step == 1:
        with st.form("sources_form"):
            uploaded = st.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
            link_text = st.text_area("Reference links or downloadable PDF/TXT links", value="\n".join(config.get("reference_links", [])), height=180)

            previous, submitted = wizard_form_nav("Save & continue to Models")
            if previous:
                load_content_state_from_config(project)
                st.session_state.config_step = max(0, st.session_state.config_step - 1)
                rerun_app()

            if submitted:
                uploaded_files = save_uploaded_sources(project, uploaded)
                references: list[str] = []
                downloaded: list[str] = uploaded_files[:]
                results: list[str] = []

                for filename in uploaded_files:
                    results.append(f"Uploaded: {filename}")

                for url in split_lines(link_text):
                    try:
                        saved_name = download_source_link(url, project)
                        if saved_name:
                            downloaded.append(saved_name)
                            results.append(f"Downloaded: {saved_name}")
                        elif "wikipedia.org/wiki/" in url.lower():
                            references.append(url)
                            results.append(f"Wikipedia reference link: {url}")
                        else:
                            references.append(url)
                            results.append(f"Reference link: {url}")
                    except Exception as exc:
                        references.append(url)
                        results.append(f"Kept as reference after download failure: {url} ({exc})")

                config["reference_links"] = list(dict.fromkeys(references))
                config["source_material"] = list(dict.fromkeys(config.get("source_material", []) + downloaded))
                config["intro_material"] = list(dict.fromkeys(config.get("intro_material", []) + downloaded))
                save_config(project, config)
                st.session_state.link_results = results
                st.session_state.config_step = min(len(steps) - 1, st.session_state.config_step + 1)
                rerun_app()

        for result in st.session_state.link_results:
            st.write(result)

        with st.expander("Current source_material folder", expanded=False):
            files = sorted(p.name for p in source_dir(project).glob("*") if p.is_file())
            st.code("\n".join(files) if files else "No files yet.", language="text")

    else:
        tts_config = config.get("_project_config", {}).get("tts_config", {})
        validator_config = config.get("_project_config", {}).get("validator_config", {})
        image_config = config.get("_project_config", {}).get("image_config", {})
        current_image_model = image_model_label_for_config(image_config.get("model", "seedream-v4"))
        image_options, current_image_model = selectbox_options_with_current(
            list(load_image_model_map().keys()),
            current_image_model,
            "seedream-v4",
        )
        current_deepseek_model = validator_config.get("model", DEFAULT_DEEPSEEK_MODEL)
        deepseek_options = list(DEEPSEEK_MODEL_CHOICES)
        if current_deepseek_model not in deepseek_options:
            current_deepseek_model = DEFAULT_DEEPSEEK_MODEL
        deepseek_model = st.selectbox(
            "Language Model",
            deepseek_options,
            index=deepseek_options.index(current_deepseek_model),
        )
        image_model = st.selectbox(
            "Image Generator Model",
            image_options,
            index=image_options.index(current_image_model),
        )
        tts_model_options = ["kokoro", "inworld-tts-1.5-max", "inworld-tts-2"]
        current_tts_model = normalized_tts_model(tts_config.get("model", "kokoro"))
        tts_model = st.selectbox(
            "Text-to-Speech Model",
            tts_model_options,
            index=tts_model_options.index(current_tts_model),
        )
        voice_options = voice_options_for_tts_model(tts_model)
        saved_voice = str(tts_config.get("voice_id", "")).strip()
        current_voice = saved_voice if saved_voice in voice_options else voice_options[0]
        voice_id = st.selectbox(
            "Voice",
            voice_options,
            index=voice_options.index(current_voice),
        )
        previous, submitted = wizard_button_nav("Save & continue to Script Generation")
        if previous:
            st.session_state.config_step = max(0, st.session_state.config_step - 1)
            rerun_app()
        if submitted:
            saved_path = save_models_page(project, config, tts_model, voice_id, deepseek_model, image_model)
            ready, _ = stage_readiness(project, "Config Wizard")
            if ready:
                set_stage_complete(project, "Config Wizard", True)
                st.session_state.pending_workspace_page = "Script Generation"
            st.session_state.config_saved_notice = f"Config saved: {saved_path}"
            rerun_app()

        st.markdown("Configuration file saved at the location below - please review and edit if needed")
        path = config_path(project)
        st.markdown(f"<div class='path-box'>{path}</div>", unsafe_allow_html=True)
        if st.session_state.config_saved_notice:
            st.success(st.session_state.config_saved_notice)
            st.session_state.config_saved_notice = ""
        if path.exists():
            modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
            st.caption(f"Last saved: {modified}")

        warnings = config_warnings(config)
        if warnings:
            st.warning(
                "This saved config still looks incomplete. Go back to the relevant step, "
                "edit it, and press that page's save button."
            )
            for warning in warnings:
                st.write(f"- {warning}")


def render_script_generation(project: str) -> None:
    st.subheader("Script Generation")
    st.markdown("Generate the section outlines and full narration script.")

    outline_status = st.container()
    if st.button("Generate Section Outlines", key="generate_section_outlines_flow", width="stretch"):
        outline_path = artifact_location(project, "outline_texts.json")
        with outline_status:
            st.write("generating section outlines")
            ok, detail = run_stage_quiet(project, "Generate Section Outlines", "generate_sections.py", "outline_texts.json")
            if not ok:
                st.error(detail)
                return
            st.write(f"section outlines generated and saved at {outline_path}.")
            st.write("revising section outlines")
            ok, detail = run_stage_quiet(project, "Revise Section Outlines", "validate_outline.py", "outline_texts.json")
            if not ok:
                st.error(detail)
                return
            st.write(f"final section outlines saved at {outline_path}.")

    narration_status = st.container()
    if st.button("Generate Full Script", key="generate_full_script_flow", width="stretch"):
        narration_path = artifact_location(project, "narration.json")
        with narration_status:
            st.write("generating full script")
            ok, detail = run_stage_quiet(project, "Generate Full Script", "generate_script.py", "narration.json")
            if not ok:
                st.error(detail)
                return
            st.write(f"full script generated and saved at {narration_path}.")
            st.write("revising full script")
            ok, detail = run_stage_quiet(project, "Revise Full Script", "validate_narration.py", "narration.json")
            if not ok:
                st.error(detail)
                return
            st.write(f"final full script saved at {narration_path}.")


def render_voice_generation(project: str) -> None:
    st.subheader("Voice Generation")

    st.markdown("Generate narration audio selectively")

    config = load_config(project)
    available_keys = narration_segment_keys(project)
    existing_audio_keys = media_keys_in_dir(output_dir(project, "audios"), "audio", (".mp3", ".wav"))
    missing_audio_keys = available_keys - existing_audio_keys
    key_options = sorted(available_keys)
    key_labels = [format_media_key(key) for key in key_options]
    key_by_label = dict(zip(key_labels, key_options))

    if not available_keys:
        st.info("Generate and validate narration first. The audio frames come from narration.json.")
        return

   
    audio_mode = st.radio(
        "Audio selection",
        ["Missing audio", "All audio", "Range", "Specific frames"],
        horizontal=True,
    )

    selected_audio_keys: set[tuple[int, int]] = set()
    audio_selection_arg = "missing"
    selection_error = ""

    if audio_mode == "Missing audio":
        selected_audio_keys = set(missing_audio_keys)
        audio_selection_arg = "missing"
        if selected_audio_keys:
            st.caption(
                "Will generate missing audio: "
                + ", ".join(format_media_key(key) for key in sorted(selected_audio_keys))
            )
        else:
            st.info("All narration frames already have audio.")
    elif audio_mode == "All audio":
        selected_audio_keys = set(available_keys)
        audio_selection_arg = "all"
    elif audio_mode == "Range":
        range_cols = st.columns(2)
        with range_cols[0]:
            start_label = st.selectbox("Start frame", key_labels, index=0)
        with range_cols[1]:
            end_label = st.selectbox("End frame", key_labels, index=len(key_labels) - 1)
        start_key = key_by_label[start_label]
        end_key = key_by_label[end_label]
        if start_key > end_key:
            start_key, end_key = end_key, start_key
        selected_audio_keys = {key for key in available_keys if start_key <= key <= end_key}
        audio_selection_arg = f"{format_media_key(start_key)}-{format_media_key(end_key)}"
    else:
        default_labels = [format_media_key(key) for key in sorted(missing_audio_keys)] or key_labels[:1]
        selected_labels = st.multiselect(
            "Specific frames",
            key_labels,
            default=default_labels,
        )
        selected_audio_keys = {key_by_label[label] for label in selected_labels}
        audio_selection_arg = ",".join(format_media_key(key) for key in sorted(selected_audio_keys))
        if not selected_audio_keys:
            selection_error = "Select at least one frame."

    if selection_error:
        st.error(selection_error)

    script = tts_script_for_config(config)
    can_generate_audio = not selection_error and bool(selected_audio_keys)
    if st.button("Generate Audio", key="run_generate_audio_selected", width="stretch", disabled=not can_generate_audio):
        run_stage(
            project,
            "Generate Audio",
            script,
            "tts_index.json",
            extra_args=["--audio-selection", audio_selection_arg],
            expected_audio_keys=selected_audio_keys,
        )

    st.metric("Audio files", len(existing_audio_keys))
    st.markdown(f"<div class='path-box'>{output_dir(project, 'audios')}</div>", unsafe_allow_html=True)


def render_video_generation(project: str) -> None:
    st.subheader("Video Generation")

    st.markdown("After image and voice generation are complete, generate segment clips and compose the final video.")

    image_keys = media_keys_in_dir(output_dir(project, "images"), "image", (".png",))
    audio_keys = media_keys_in_dir(output_dir(project, "audios"), "audio", (".mp3", ".wav"))
    matched_keys = set(sorted(image_keys & audio_keys))
    existing_clip_keys = media_keys_in_dir(output_dir(project, "clips"), "image", (".mp4",))
    missing_clip_keys = matched_keys - existing_clip_keys
    key_options = sorted(matched_keys)
    key_labels = [format_media_key(key) for key in key_options]
    key_by_label = dict(zip(key_labels, key_options))

    if image_keys - audio_keys:
        st.warning(
            "Some approved images do not have matching audio: "
            + ", ".join(format_media_key(key) for key in sorted(image_keys - audio_keys))
        )
    if audio_keys - image_keys:
        st.warning(
            "Some audio files do not have matching approved images: "
            + ", ".join(format_media_key(key) for key in sorted(audio_keys - image_keys))
        )

    clip_mode = st.radio(
        "Clip selection",
        ["Missing clips", "All clips", "Range", "Specific clips"],
        horizontal=True,
    )

    selected_clip_keys: set[tuple[int, int]] = set()
    clip_selection_arg = "missing"
    selection_error = ""

    if clip_mode == "Missing clips":
        selected_clip_keys = set(missing_clip_keys)
        clip_selection_arg = "missing"
        if selected_clip_keys:
            st.caption(
                "Will generate missing clips: "
                + ", ".join(format_media_key(key) for key in sorted(selected_clip_keys))
            )
        else:
            st.info("All matched clips already exist.")
    elif clip_mode == "All clips":
        selected_clip_keys = set(matched_keys)
        clip_selection_arg = "all"
    elif clip_mode == "Range":
        if key_options:
            range_cols = st.columns(2)
            with range_cols[0]:
                start_label = st.selectbox("Start clip", key_labels, index=0)
            with range_cols[1]:
                end_label = st.selectbox("End clip", key_labels, index=len(key_labels) - 1)
            start_key = key_by_label[start_label]
            end_key = key_by_label[end_label]
            if start_key > end_key:
                start_key, end_key = end_key, start_key
            selected_clip_keys = {key for key in matched_keys if start_key <= key <= end_key}
            clip_selection_arg = f"{format_media_key(start_key)}-{format_media_key(end_key)}"
        else:
            selection_error = "No matched image/audio pairs are available."
    else:
        default_labels = [format_media_key(key) for key in sorted(missing_clip_keys)] or key_labels[:1]
        selected_labels = st.multiselect(
            "Specific clips",
            key_labels,
            default=default_labels,
        )
        selected_clip_keys = {key_by_label[label] for label in selected_labels}
        clip_selection_arg = ",".join(format_media_key(key) for key in sorted(selected_clip_keys))
        if not selected_clip_keys:
            selection_error = "Select at least one clip."

    if selection_error:
        st.error(selection_error)

    can_generate_clips = bool(matched_keys) and not selection_error and bool(selected_clip_keys)
    if st.button("Generate Clips", key="run_generate_clips_selected", width="stretch", disabled=not can_generate_clips):
        run_stage(
            project,
            "Generate Clips",
            "generate_clips.py",
            "clips",
            extra_args=["--clip-selection", clip_selection_arg],
            expected_clip_keys=selected_clip_keys,
        )

    if st.button("Compose Final Video", key="run_make_video.py", width="stretch"):
        if compose_final_video(project):
            set_stage_complete(project, "Video Generation", True)


def render_image_generation(project: str) -> None:
    st.subheader("Image Generation")
    
    config = load_config(project)

    label, script, expected = IMAGE_PROMPT_STAGE
    st.markdown(f"<div class='stage-row'><strong>{label}</strong></div>", unsafe_allow_html=True)
    if st.button(label, key=f"run_{script}_{label}", width="stretch"):
        run_stage(project, label, script, expected)

    
    items = load_image_prompt_items(project)
    if not items:
        st.info("Generate Image Prompts first. The app will then show each prompt here.")
        return

    st.markdown(f"<div class='stage-row'><strong>Generate Images</strong></div>", unsafe_allow_html=True)
    image_generation_mode = st.selectbox(
        "Image generation mode",
        ["Review images one by one", "Generate all images one by one without review"],
    )
    if image_generation_mode == "Generate all images one by one without review":
        if st.button("Generate all images without review", key="generate_all_images_no_review", width="stretch"):
            if generate_all_images_without_review(project):
                set_stage_complete(project, "Image Generation", True)
        return

    st.session_state.active_review_index = max(0, min(st.session_state.active_review_index, len(items) - 1))

    nav_prev, nav_jump, nav_next = st.columns([1, 3, 1])
    with nav_prev:
        if st.button("Previous prompt", disabled=st.session_state.active_review_index == 0, width="stretch"):
            set_active_review_index(st.session_state.active_review_index - 1, len(items))
            rerun_app()
    with nav_jump:
        selected_index = st.selectbox(
            "Jump to prompt",
            options=list(range(len(items))),
            index=st.session_state.active_review_index,
            format_func=lambda index: (
                f"{index + 1}. Section {items[index]['section_index']}."
                f"{items[index]['prompt_index']} - {items[index]['section_title']}"
            ),
        )
        if selected_index != st.session_state.active_review_index:
            set_active_review_index(selected_index, len(items))
            rerun_app()
    with nav_next:
        if st.button("Next prompt", disabled=st.session_state.active_review_index >= len(items) - 1, width="stretch"):
            set_active_review_index(st.session_state.active_review_index + 1, len(items))
            rerun_app()

    item = items[st.session_state.active_review_index]
    section_index = item["section_index"]
    prompt_index = item["prompt_index"]

    approved_path = output_dir(project, "images") / f"image_{section_index}_{prompt_index}.png"
    review_dir = output_dir(project, "images") / "_review"
    rejected_dir = output_dir(project, "rejected_images")
    active_review_path = Path(st.session_state.active_review_path) if st.session_state.active_review_path else None
    has_review_image = bool(active_review_path and active_review_path.exists())

    left, right = st.columns([2, 1])
    with left:
        st.caption(f"{st.session_state.active_review_index + 1} of {len(items)}")
        st.markdown(f"**Section {section_index}.{prompt_index}: {item['section_title']}**")
    with right:
        st.progress((st.session_state.active_review_index + 1) / len(items))

    if st.session_state.active_prompt_text == "":
        st.session_state.active_prompt_text = item["prompt"]

    if st.session_state.image_review_notice:
        notice_kind = st.session_state.image_review_notice_kind
        if notice_kind == "warning":
            st.warning(st.session_state.image_review_notice)
        elif notice_kind == "error":
            st.error(st.session_state.image_review_notice)
        else:
            st.success(st.session_state.image_review_notice)
        st.session_state.image_review_notice = ""
        st.session_state.image_review_notice_kind = "success"

    st.text_area("Narration segment", value=item["narration"], height=120, disabled=True)
    prompt_text = st.text_area("Image prompt", value=st.session_state.active_prompt_text, height=180)
    st.session_state.active_prompt_text = prompt_text

    cols = st.columns([1, 1, 1])
    with cols[0]:
        generate = st.button("Generate new image", width="stretch")
    with cols[1]:
        keep = st.button("Keep image", disabled=not has_review_image, width="stretch")
    with cols[2]:
        reject = st.button("Reject image", disabled=not has_review_image, width="stretch")

    if generate:
        try:
            with st.spinner("Generating image..."):
                image = generate_image_with_fal(prompt_text, config)
                attempt = int(time.time())
                review_path = review_dir / f"image_{section_index}_{prompt_index}_attempt{attempt}.png"
                save_image_checked(image, review_path)
                st.session_state.active_review_path = str(review_path)
            st.session_state.image_review_notice = f"Review image saved: {review_path}"
            st.session_state.image_review_notice_kind = "success"
            rerun_app()
        except Exception as exc:
            st.error(str(exc))

    if has_review_image:
        st.image(str(active_review_path), caption="Current review image", width="stretch")
    elif approved_path.exists():
        st.image(str(approved_path), caption="Approved image already exists", width="stretch")

    if keep and st.session_state.active_review_path:
        try:
            copy_checked(Path(st.session_state.active_review_path), approved_path)
            st.session_state.image_review_notice = f"Approved image saved: {approved_path}"
            st.session_state.image_review_notice_kind = "success"
            st.session_state.active_review_path = ""
            st.session_state.active_prompt_text = ""
            st.session_state.active_review_index = min(len(items) - 1, st.session_state.active_review_index + 1)
            rerun_app()
        except Exception as exc:
            st.error(str(exc))

    if reject and st.session_state.active_review_path:
        try:
            existing = sorted(rejected_dir.glob(f"image_{section_index}_{prompt_index}_v*.png"))
            reject_path = rejected_dir / f"image_{section_index}_{prompt_index}_v{len(existing) + 1}.png"
            copy_checked(Path(st.session_state.active_review_path), reject_path)
            st.session_state.image_review_notice = f"Rejected image saved: {reject_path}"
            st.session_state.image_review_notice_kind = "warning"
            st.session_state.active_review_path = ""
            rerun_app()
        except Exception as exc:
            st.error(str(exc))

    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        if approved_path.exists():
            st.caption("This prompt already has an approved image. Generate a new image here to replace it after review.")
        else:
            st.caption("Generate an image, then keep or reject it.")
    with nav_right:
        approved_count = len(list(output_dir(project, "images").glob("image_*.png")))
        st.metric("Approved images", approved_count)


def generate_all_images_without_review(project: str) -> bool:
    config = load_config(project)
    items = load_image_prompt_items(project)
    if not items:
        st.error("Image prompts are missing. Generate image_prompts.json first.")
        return False

    progress_text = st.empty()
    progress_bar = st.progress(0.0)
    try:
        for index, item in enumerate(items, start=1):
            section_index = item["section_index"]
            prompt_index = item["prompt_index"]
            progress_text.info(f"Generating image {index} of {len(items)}")
            image = generate_image_with_fal(item["prompt"], config)
            approved_path = output_dir(project, "images") / f"image_{section_index}_{prompt_index}.png"
            save_image_checked(image, approved_path)
            progress_bar.progress(index / len(items))
    except Exception as exc:
        st.error(f"Automatic image generation failed: {exc}")
        return False
    finally:
        time.sleep(0.2)
        progress_text.empty()
        progress_bar.empty()

    st.success("Images generated automatically.")
    st.markdown(f"<div class='path-box'>{output_dir(project, 'images')}</div>", unsafe_allow_html=True)
    return True


def run_full_automatic_pipeline(project: str) -> None:
    st.warning("Automatic mode skips image review and saves generated images directly as approved images.")

    if config_warnings(load_config(project)):
        st.error("Config is incomplete. Save the required config fields before running automatic mode.")
        return

    set_stage_complete(project, "Config Wizard", True)

    for label, script, expected in SCRIPT_STAGES:
        if not run_stage(project, label, script, expected):
            return
    set_stage_complete(project, "Script Generation", True)

    label, script, expected = IMAGE_PROMPT_STAGE
    if not run_stage(project, label, script, expected):
        return

    if not generate_all_images_without_review(project):
        return
    set_stage_complete(project, "Image Generation", True)

    config = load_config(project)
    audio_keys = narration_segment_keys(project)
    if not run_stage(
        project,
        "Generate Audio",
        tts_script_for_config(config),
        "tts_index.json",
        extra_args=["--audio-selection", "all"],
        expected_audio_keys=audio_keys,
    ):
        return
    set_stage_complete(project, "Voice Generation", True)

    image_keys = media_keys_in_dir(output_dir(project, "images"), "image", (".png",))
    audio_file_keys = media_keys_in_dir(output_dir(project, "audios"), "audio", (".mp3", ".wav"))
    missing_audio = sorted(image_keys - audio_file_keys)
    missing_images = sorted(audio_file_keys - image_keys)
    if missing_audio or missing_images:
        if missing_audio:
            st.error(
                "Automatic mode stopped before clip generation. Missing audio for: "
                + ", ".join(format_media_key(key) for key in missing_audio[:12])
            )
        if missing_images:
            st.error(
                "Automatic mode stopped before clip generation. Missing images for: "
                + ", ".join(format_media_key(key) for key in missing_images[:12])
            )
        return

    clip_keys = image_keys
    if not run_stage(
        project,
        "Generate Clips",
        "generate_clips.py",
        "clips",
        extra_args=["--clip-selection", "all"],
        expected_clip_keys=clip_keys,
    ):
        return

    if not compose_final_video(project):
        return
    set_stage_complete(project, "Video Generation", True)


def render_page_footer(project: str, page: str) -> None:
    if page not in WORKFLOW_PAGES:
        return
    if page == "Video Generation":
        return

    st.markdown("---")
    if page == "Config Wizard":
        auto_disabled = bool(config_warnings(load_config(project)))
        if st.button(
            "Generate Video in Fully Automated Mode",
            key="run_full_auto_pipeline",
            width="stretch",
            disabled=auto_disabled,
            type="primary",
        ):
            run_full_automatic_pipeline(project)
        if auto_disabled:
            st.caption("Complete and save the config before running the automatic pipeline.")
        return

    left, right = st.columns([1, 1])
    ready_to_mark, readiness_message = stage_readiness(project, page)
    is_acknowledged = stage_acknowledged(project, page)

    with left:
        mark_label = "MARK STAGE AS INCOMPLETE" if is_acknowledged else "MARK STAGE AS COMPLETE"
        mark_disabled = not is_acknowledged and not ready_to_mark
        if st.button(
            mark_label,
            key=f"mark_complete_{page}",
            width="stretch",
            disabled=mark_disabled,
        ):
            set_stage_complete(project, page, not is_acknowledged)
            rerun_app()
        if not ready_to_mark and not is_acknowledged:
            st.caption(readiness_message)

    current_index = WORKFLOW_PAGES.index(page)
    if current_index < len(WORKFLOW_PAGES) - 1:
        next_page = WORKFLOW_PAGES[current_index + 1]
        next_disabled = not stage_acknowledged(project, page)
        with right:
            if st.button(
                f"Go to {next_page}",
                key=f"go_to_{next_page}",
                width="stretch",
                disabled=next_disabled,
            ):
                st.session_state.pending_workspace_page = next_page
                rerun_app()


def render_outputs(project: str) -> None:
    st.subheader("Outputs")
    for label, folder in OUTPUT_PAGE_FOLDERS:
        path = output_dir(project, folder)
        st.markdown(f"**{label}**")
        st.markdown(f"<div class='path-box'>{path}</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Agentic Video Generator", layout="wide")
    inject_css()
    init_state()

    project = render_project_selector()
    if not project:
        render_header()
        st.info("Create or select a project to begin.")
        return

    ensure_project_dirs(project)

    st.sidebar.divider()
    pages = WORKFLOW_PAGES + ["Outputs"]
    query_page = selected_stage_from_query()
    if query_page in pages:
        st.session_state.workspace_page = query_page

    pending_page = st.session_state.get("pending_workspace_page", "")
    if pending_page in pages:
        st.session_state.workspace_page = pending_page
        try:
            st.query_params["stage"] = pending_page
        except Exception:
            pass
    st.session_state.pending_workspace_page = ""
    if st.session_state.workspace_page not in pages:
        st.session_state.workspace_page = "Config Wizard"

    page = render_sidebar_navigation(project, pages)

    render_header(page)
    st.markdown(f"Project: **{project}**")

    if page == "Config Wizard":
        render_config_wizard(project)
    elif page == "Script Generation":
        render_script_generation(project)
    elif page == "Image Generation":
        render_image_generation(project)
    elif page == "Voice Generation":
        render_voice_generation(project)
    elif page == "Video Generation":
        render_video_generation(project)
    else:
        render_outputs(project)

    render_page_footer(project, page)


def running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def relaunch_with_streamlit() -> int:
    command = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
    command.extend(sys.argv[1:])
    print("Starting Streamlit app...")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    return subprocess.run(command).returncode


if __name__ == "__main__":
    if running_inside_streamlit():
        main()
    else:
        raise SystemExit(relaunch_with_streamlit())
