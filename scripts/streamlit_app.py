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
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
import streamlit as st
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

IMAGE_MODEL_MAP = {
    "seedream-v4": "fal-ai/bytedance/seedream/v4/text-to-image",
    "seedream_v4": "fal-ai/bytedance/seedream/v4/text-to-image",
}

NARRATION_STAGES = [
    ("Generate Sections", "generate_sections.py", "outline_texts.json"),
    ("Validate Outline", "validate_outline.py", "outline_texts.json"),
    ("Generate Narration", "generate_script.py", "narration.json"),
    ("Validate Narration", "validate_narration.py", "narration.json"),
    ("Generate Image Prompts", "generate_image_prompts.py", "image_prompts.json"),
    ("Generate Audio", None, "tts_index.json"),
]

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
            --accent: #6d28d9;
            --accent-2: #be123c;
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
            padding-top: 2rem;
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
            padding: 22px 24px;
            background:
                linear-gradient(135deg, rgba(32, 227, 178, .24) 0%, rgba(76, 201, 240, .18) 52%, rgba(255, 209, 102, .12) 100%);
            margin-bottom: 18px;
            box-shadow: 0 16px 42px rgba(0, 0, 0, .25);
        }

        .app-hero h1 {
            margin: 0;
            font-size: 2.617rem;
            line-height: 1.15;
        }

        .app-hero p {
            color: var(--muted);
            margin: 8px 0 0 0;
            max-width: 760px;
            font-size: 1.287rem;
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

        .ok-pill, .wait-pill {
            border-radius: 999px;
            padding: 5px 11px;
            font-size: 1.067rem;
            font-weight: 800;
            display: inline-block;
        }

        .ok-pill {
            background: var(--accent-soft);
            color: #7dffd8;
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

        .wait-pill {
            background: var(--warn-soft);
            color: #ffe39a;
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
            "narration_config": {"words_per_section": 400},
            "validator_config": {"model": "deepseek-chat"},
            "tts_config": {"model": "kokoro", "voice_id": "af"},
        },
    }


def ensure_config_defaults(config: dict, project: str) -> dict:
    project_config = config.setdefault("_project_config", {})
    project_config.setdefault("project_name", project)
    if not isinstance(project_config.get("image_config"), dict):
        project_config["image_config"] = {"model": "seedream-v4"}
    if not isinstance(project_config.get("narration_config"), dict):
        project_config["narration_config"] = {"words_per_section": 400}
    if not isinstance(project_config.get("validator_config"), dict):
        project_config["validator_config"] = {"model": "deepseek-chat"}
    if not isinstance(project_config.get("tts_config"), dict):
        project_config["tts_config"] = {"model": "kokoro", "voice_id": "af"}
    project_config["image_config"].setdefault("model", "seedream-v4")
    project_config["narration_config"].setdefault("words_per_section", 400)
    project_config["validator_config"].setdefault("model", "deepseek-chat")
    project_config["tts_config"].setdefault("model", "kokoro")
    project_config["tts_config"].setdefault("voice_id", "af")
    return config


def config_path(project: str) -> Path:
    return source_dir(project) / "config.json"


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
    return {
        "outline_texts.json": (base / "outline_texts.json").exists(),
        "narration.json": (base / "narration.json").exists(),
        "image_prompts.json": (base / "image_prompts.json").exists(),
        "tts_index.json": (base / "tts_index.json").exists(),
        "video": any(output_dir(project, "videos").glob("*.mp4")),
    }


def config_warnings(config: dict) -> list[str]:
    warnings: list[str] = []
    if not config.get("video_title", "").strip() or config.get("video_title") == "Your Video Title Here":
        warnings.append("Video title is empty or still a template value.")
    section_outlines = config.get("section_outlines", [])
    if not section_outlines:
        warnings.append("No section outlines are saved.")
    elif any("Section 1: Opening and introduction" in str(item) for item in section_outlines):
        warnings.append("Section outlines still look like template defaults.")
    if not config.get("narration_style", []):
        warnings.append("No narration style rules are saved.")
    if not config.get("historical_context", "").strip():
        warnings.append("Historical or subject context is empty.")
    if not config.get("aesthetic_style", "").strip():
        warnings.append("Aesthetic style is empty.")
    if not config.get("_project_config", {}).get("project_name"):
        warnings.append("Project name is missing from _project_config.")
    return warnings


def progress_event_from_line(line: str) -> tuple[str, int, int] | None:
    patterns = [
        (
            r"Generating narration section\s+(\d+)\s*/\s*(\d+)",
            "Generating narration section {current} of {total}",
        ),
        (
            r"Generating image prompt\s+(\d+)\s*/\s*(\d+)",
            "Generating image prompt {current} of {total}",
        ),
        (
            r"Generating clip\s+(\d+)\s*/\s*(\d+)",
            "Generating clip {current} of {total}",
        ),
        (
            r"Loading final video clip\s+(\d+)\s*/\s*(\d+)",
            "Loading final video clip {current} of {total}",
        ),
        (
            r"Writing final video\s+(\d+)\s*/\s*(\d+)",
            "Writing final video {current} of {total}",
        ),
    ]
    for pattern, template in patterns:
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            current = int(match.group(1))
            total = max(1, int(match.group(2)))
            return template.format(current=current, total=total), current, total
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

    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        lines.append(line)
        progress_event = progress_event_from_line(line)
        if progress_event:
            message, current, total = progress_event
            saw_progress = True
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
    if expected.endswith(".json"):
        return output_dir(project, "output_jsons") / expected
    if expected == "clips":
        return output_dir(project, "clips")
    if expected == "videos":
        return output_dir(project, "videos")
    return None


def stage_success_text(label: str, expected: str | None) -> str:
    if expected == "outline_texts.json":
        return "Outline generated." if "Generate" in label else "Outline validated."
    if expected == "narration.json":
        return "Narration generated." if "Generate" in label else "Narration validated."
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


def parse_media_key_text(value: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d+)[_\-:](\d+)\s*$", value)
    if not match:
        raise ValueError(f"Invalid clip key: {value}. Use section_segment, e.g. 3_2.")
    return int(match.group(1)), int(match.group(2))


def parse_clip_selection_text(selection: str, available_keys: set[tuple[int, int]]) -> set[tuple[int, int]]:
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
        raise ValueError(f"Requested clips do not have matching image/audio pairs: {missing_text}")
    return requested


def verify_stage_artifact(
    project: str,
    expected: str | None,
    expected_clip_keys: set[tuple[int, int]] | None = None,
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
        path = output_dir(project, "videos")
        if not any(path.glob("*.mp4")):
            return False, f"No final MP4 files found in: {path}"
        return True, f"Verified video in: {path}"

    return True, ""


def tts_script_for_config(config: dict) -> str:
    model = (
        config.get("_project_config", {})
        .get("tts_config", {})
        .get("model", "kokoro")
        .strip()
        .lower()
    )
    if model == "inworld":
        return "generate_inworld_voice.py"
    return "generate_kokoro_voice.py"


def run_stage(
    project: str,
    label: str,
    script_name: str,
    expected_output: str | None = None,
    extra_args: list[str] | None = None,
    expected_clip_keys: set[tuple[int, int]] | None = None,
) -> None:
    with st.spinner(f"Running {label}..."):
        code, log = run_script(script_name, project, extra_args=extra_args)
    if code == 0:
        ok, message = verify_stage_artifact(
            project,
            expected_output,
            expected_clip_keys=expected_clip_keys,
        )
        if ok:
            location = artifact_location(project, expected_output)
            st.success(stage_success_text(label, expected_output))
            if location:
                st.markdown(f"<div class='path-box'>{location}</div>", unsafe_allow_html=True)
        else:
            st.error(f"{label} finished but output validation failed.")
            st.write(message)
            recent_log = "\n".join(log.splitlines()[-40:])
            with st.expander("Recent log output", expanded=False):
                st.code(recent_log or "No log output captured.", language="text")
    else:
        st.error(f"{label} failed with exit code {code}.")
        recent_log = "\n".join(log.splitlines()[-40:])
        with st.expander("Recent log output", expanded=False):
            st.code(recent_log or "No log output captured.", language="text")


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
    model_key = (
        config.get("_project_config", {})
        .get("image_config", {})
        .get("model", "seedream-v4")
        .strip()
        .lower()
    )
    return IMAGE_MODEL_MAP.get(model_key, model_key)


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


def render_header() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <h1>Agentic Video Generator</h1>
            <p>Build the project config, generate narration and audio, review images, and compose the final video from one guided workspace.</p>
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


def render_project_metrics(project: str) -> None:
    status = output_status(project)
    cols = st.columns(5)
    labels = [
        ("Outline", status["outline_texts.json"]),
        ("Narration", status["narration.json"]),
        ("Prompts", status["image_prompts.json"]),
        ("Audio Index", status["tts_index.json"]),
        ("Video", status["video"]),
    ]
    for col, (label, ok) in zip(cols, labels):
        with col:
            pill = "Ready" if ok else "Pending"
            klass = "ok-pill" if ok else "wait-pill"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="{klass}">{pill}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def step_controls(total_steps: int) -> None:
    st.caption("Use the step's Save and continue button to write changes before moving forward.")
    if st.button("Previous", disabled=st.session_state.config_step == 0):
        st.session_state.config_step = max(0, st.session_state.config_step - 1)
        rerun_app()


def render_config_wizard(project: str) -> None:
    config = load_config(project)
    steps = ["Basics", "Outline", "Sources", "Voice", "Visuals", "Review"]

    st.subheader("Configuration Wizard")
    st.progress((st.session_state.config_step + 1) / len(steps))
    st.caption(f"Step {st.session_state.config_step + 1} of {len(steps)}: {steps[st.session_state.config_step]}")

    step = st.session_state.config_step

    if step == 0:
        with st.form("basics_form"):
            title = st.text_input("Video title", value=config.get("video_title", ""))
            n_section = st.number_input("Number of sections", min_value=1, max_value=30, value=int(config.get("n_section", 6)))
            historical_context = st.text_area("Historical or subject context", value=config.get("historical_context", ""), height=110)

            submitted = st.form_submit_button("Save basics and continue", width="stretch")
            if submitted:
                config["video_title"] = title.strip()
                config["n_section"] = int(n_section)
                config["historical_context"] = historical_context.strip()
                config["_project_config"]["project_name"] = project
                save_config(project, config)
                st.success("Basics saved.")
                st.session_state.config_step = min(len(steps) - 1, st.session_state.config_step + 1)
                rerun_app()

    elif step == 1:
        outlines = config.get("section_outlines", [])
        outline_text = "\n".join(outlines)
        with st.form("outline_form"):
            st.caption("Write one section guideline per line. The generation script will turn these into structured sections.")
            updated = st.text_area("Section guidelines", value=outline_text, height=340)
            submitted = st.form_submit_button("Save outline and continue", width="stretch")
            if submitted:
                config["section_outlines"] = split_lines(updated)
                config["n_section"] = len(config["section_outlines"])
                save_config(project, config)
                st.success(f"Saved {len(config['section_outlines'])} section guidelines.")
                st.session_state.config_step = min(len(steps) - 1, st.session_state.config_step + 1)
                rerun_app()

    elif step == 2:
        st.caption("PDF and TXT links are downloaded into source_material. HTML and Wikipedia links stay as reference links for retrieval.")
        uploaded = st.file_uploader("Upload PDFs or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
        if st.button("Save uploaded files", width="stretch"):
            saved = save_uploaded_sources(project, uploaded)
            material = list(dict.fromkeys(config.get("source_material", []) + saved))
            config["source_material"] = material
            config["intro_material"] = list(dict.fromkeys(config.get("intro_material", []) + saved))
            save_config(project, config)
            st.success(f"Saved {len(saved)} uploaded source files.")

        link_text = st.text_area("Reference links or downloadable PDF/TXT links", value="\n".join(config.get("reference_links", [])), height=180)
        if st.button("Process links and save", width="stretch"):
            references: list[str] = []
            downloaded: list[str] = []
            results: list[str] = []

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
            st.success("Links processed.")

        for result in st.session_state.link_results:
            st.write(result)

        with st.expander("Current source_material folder", expanded=False):
            files = sorted(p.name for p in source_dir(project).glob("*") if p.is_file())
            st.code("\n".join(files) if files else "No files yet.", language="text")

        if st.button("Save source links and continue", width="stretch"):
            config["reference_links"] = list(dict.fromkeys(split_lines(link_text)))
            save_config(project, config)
            st.session_state.config_step = min(len(steps) - 1, st.session_state.config_step + 1)
            rerun_app()

    elif step == 3:
        tts_config = config.get("_project_config", {}).get("tts_config", {})
        validator_config = config.get("_project_config", {}).get("validator_config", {})
        narration_config = config.get("_project_config", {}).get("narration_config", {})
        with st.form("voice_form"):
            narration_style = st.text_area(
                "Narration style rules",
                value="\n".join(config.get("narration_style", [])),
                height=190,
            )
            words_per_section = st.number_input(
                "Tentative words per section",
                min_value=100,
                max_value=3000,
                step=50,
                value=int(narration_config.get("words_per_section", 400)),
            )
            tts_model = st.selectbox(
                "TTS model",
                ["kokoro", "inworld"],
                index=0 if tts_config.get("model", "kokoro").lower() != "inworld" else 1,
            )
            voice_id = st.text_input("Voice ID", value=tts_config.get("voice_id", "af"))
            validator_model = st.text_input(
                "DeepSeek validator model",
                value=validator_config.get("model", "deepseek-chat"),
            )
            submitted = st.form_submit_button("Save voice settings and continue", width="stretch")
            if submitted:
                config["narration_style"] = split_lines(narration_style)
                config["_project_config"]["narration_config"] = {
                    "words_per_section": int(words_per_section)
                }
                config["_project_config"]["tts_config"] = {"model": tts_model, "voice_id": voice_id.strip()}
                config["_project_config"]["validator_config"] = {
                    "model": validator_model.strip() or "deepseek-chat"
                }
                save_config(project, config)
                st.success("Voice settings saved.")
                st.session_state.config_step = min(len(steps) - 1, st.session_state.config_step + 1)
                rerun_app()

    elif step == 4:
        image_config = config.get("_project_config", {}).get("image_config", {})
        with st.form("visual_form"):
            aesthetic_style = st.text_area("Aesthetic style", value=config.get("aesthetic_style", ""), height=120)
            characters = st.text_area(
                "Character canon, one per line as Name: visual description",
                value=characters_to_text(config.get("characters", {})),
                height=220,
            )
            image_model = st.text_input("Image model", value=image_config.get("model", "seedream-v4"))
            submitted = st.form_submit_button("Save visual settings and continue", width="stretch")
            if submitted:
                config["aesthetic_style"] = aesthetic_style.strip()
                config["characters"] = parse_characters(characters)
                config["_project_config"]["image_config"] = {"model": image_model.strip() or "seedream-v4"}
                save_config(project, config)
                st.success("Visual settings saved.")
                st.session_state.config_step = min(len(steps) - 1, st.session_state.config_step + 1)
                rerun_app()

    else:
        st.markdown("Saved configuration file")
        path = config_path(project)
        st.markdown(f"<div class='path-box'>{path}</div>", unsafe_allow_html=True)
        if path.exists():
            modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
            st.caption(f"Last saved: {modified}")

        warnings = config_warnings(config)
        if warnings:
            st.warning(
                "This saved config still looks incomplete. Go back to the relevant step, "
                "edit it, and press that step's Save and continue button."
            )
            for warning in warnings:
                st.write(f"- {warning}")
        else:
            st.success("Saved config looks complete enough to run narration.")

        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Sections", len(config.get("section_outlines", [])))
        with summary_cols[1]:
            st.metric("Reference links", len(config.get("reference_links", [])))
        with summary_cols[2]:
            st.metric("Source files", len(config.get("source_material", [])))
        with summary_cols[3]:
            st.metric("Characters", len(config.get("characters", {})))

        st.caption("This page shows a concise saved-config summary. The full config.json is available at the path above.")
        if st.button("Reload from disk", width="stretch"):
            rerun_app()

    step_controls(len(steps))


def render_narration(project: str) -> None:
    st.subheader("Narration")
    config = load_config(project)

    st.markdown("Generate the outline, narration, image prompts, and audio files.")
    for label, script, expected in NARRATION_STAGES:
        actual_script = script or tts_script_for_config(config)
        st.markdown(f"<div class='stage-row'><strong>{label}</strong></div>", unsafe_allow_html=True)
        if st.button(f"Run {label}", key=f"run_{actual_script}_{label}", width="stretch"):
            run_stage(project, label, actual_script, expected)


def render_video_generation(project: str) -> None:
    st.subheader("Video Generation")

    st.markdown("After image review is complete, generate segment clips and compose the final video.")

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

    st.markdown("<div class='stage-row'><strong>Generate Clips</strong></div>", unsafe_allow_html=True)
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
    if st.button("Run Generate Clips", key="run_generate_clips_selected", width="stretch", disabled=not can_generate_clips):
        run_stage(
            project,
            "Generate Clips",
            "generate_clips.py",
            "clips",
            extra_args=["--clip-selection", clip_selection_arg],
            expected_clip_keys=selected_clip_keys,
        )

    st.markdown("<div class='stage-row'><strong>Compose Final Video</strong></div>", unsafe_allow_html=True)
    if st.button("Run Compose Final Video", key="run_make_video.py", width="stretch"):
        run_stage(project, "Compose Final Video", "make_video.py", "videos")


def render_image_review(project: str) -> None:
    st.subheader("Image Review")
    config = load_config(project)
    items = load_image_prompt_items(project)
    if not items:
        st.info("Run Generate Image Prompts first. The app will then show each prompt here.")
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


def render_outputs(project: str) -> None:
    st.subheader("Outputs")
    for folder in OUTPUT_FOLDERS:
        path = output_dir(project, folder)
        st.markdown(f"**{folder}**")
        st.markdown(f"<div class='path-box'>{path}</div>", unsafe_allow_html=True)
        files = sorted([p.name for p in path.glob("*") if p.is_file()])[:40]
        if files:
            st.code("\n".join(files), language="text")
        else:
            st.caption("No files yet.")


def main() -> None:
    st.set_page_config(page_title="Agentic Video Generator", layout="wide")
    inject_css()
    init_state()
    render_header()

    project = render_project_selector()
    if not project:
        st.info("Create or select a project to begin.")
        return

    ensure_project_dirs(project)
    render_project_metrics(project)

    st.sidebar.divider()
    page = st.sidebar.radio(
        "Workspace",
        ["Config Wizard", "Narration", "Image Review", "Video Generation", "Outputs"],
        index=0,
    )

    st.markdown(f"Project: **{project}**")
    st.markdown(f"<div class='path-box'>{project_root(project)}</div>", unsafe_allow_html=True)

    if page == "Config Wizard":
        render_config_wizard(project)
    elif page == "Narration":
        render_narration(project)
    elif page == "Image Review":
        render_image_review(project)
    elif page == "Video Generation":
        render_video_generation(project)
    else:
        render_outputs(project)


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
