"""Streamlit app for the agentic video generator pipeline.

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

PIPELINE_STAGES = [
    ("Generate Sections", "generate_sections.py"),
    ("Validate Outline", "validate_outline.py"),
    ("Generate Narration", "generate_script.py"),
    ("Validate Narration", "validate_narration.py"),
    ("Generate Image Prompts", "generate_image_prompts.py"),
]

FINISH_STAGES = [
    ("Generate Audio", None),
    ("Generate Clips", "generate_clips.py"),
    ("Compose Final Video", "make_video.py"),
]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: #ffffff;
            --ink: #14213d;
            --muted: #58708a;
            --line: #d8e5f2;
            --accent: #0077b6;
            --accent-2: #00a896;
            --accent-soft: #e0f7fa;
            --warn-soft: #fff3c4;
            --page: #f4fbff;
        }

        .stApp {
            background:
                linear-gradient(180deg, #f0fbff 0%, #ffffff 46%, #f7fbff 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1180px;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        .app-hero {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 22px 24px;
            background:
                linear-gradient(135deg, rgba(0, 168, 150, .16) 0%, rgba(0, 119, 182, .10) 45%, #ffffff 100%);
            margin-bottom: 18px;
            box-shadow: 0 10px 30px rgba(20, 33, 61, .06);
        }

        .app-hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
        }

        .app-hero p {
            color: var(--muted);
            margin: 8px 0 0 0;
            max-width: 760px;
        }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            background: var(--surface);
            box-shadow: 0 6px 18px rgba(20, 33, 61, .05);
        }

        .metric-label {
            color: var(--muted);
            font-size: .82rem;
            text-transform: uppercase;
            letter-spacing: .04em;
        }

        .metric-value {
            color: var(--ink);
            font-weight: 700;
            font-size: 1.2rem;
            margin-top: 4px;
        }

        .path-box {
            background: #f6f8fa;
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 10px 12px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: .84rem;
            word-break: break-all;
        }

        .stage-row {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            background: var(--surface);
            margin-bottom: 10px;
            box-shadow: 0 6px 18px rgba(20, 33, 61, .04);
        }

        .ok-pill, .wait-pill {
            border-radius: 999px;
            padding: 3px 9px;
            font-size: .8rem;
            display: inline-block;
        }

        .ok-pill {
            background: var(--accent-soft);
            color: var(--accent);
        }

        div.stButton > button[kind="primary"],
        div.stFormSubmitButton > button {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: #ffffff;
            border: 0;
        }

        .wait-pill {
            background: var(--warn-soft);
            color: #8a5a00;
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
            "validator_config": {"model": "deepseek-chat"},
            "tts_config": {"model": "kokoro", "voice_id": "af"},
        },
    }


def config_path(project: str) -> Path:
    return source_dir(project) / "config.json"


def load_config(project: str) -> dict:
    path = config_path(project)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("_project_config", {}).setdefault("project_name", project)
        return data
    return default_config(project)


def save_config(project: str, config: dict) -> Path:
    ensure_project_dirs(project)
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


def run_script(script_name: str, project: str) -> tuple[int, str]:
    script_path = SCRIPT_DIR / script_name
    process = subprocess.Popen(
        [sys.executable, str(script_path), project],
        cwd=str(SCRIPT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    lines: list[str] = []
    log_box = st.empty()

    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        log_box.code("\n".join(lines[-120:]), language="text")

    return_code = process.wait()
    full_log = "\n".join(lines)
    return return_code, full_log


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


def run_stage(project: str, label: str, script_name: str) -> None:
    with st.spinner(f"Running {label}..."):
        code, log = run_script(script_name, project)
    if code == 0:
        st.success(f"{label} completed.")
    else:
        st.error(f"{label} failed with exit code {code}.")
        with st.expander("Full log", expanded=True):
            st.code(log, language="text")


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

    logs: list[str] = []
    log_box = st.empty()

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                message = log.get("message", "")
                if message:
                    logs.append(message)
                    log_box.code("\n".join(logs[-20:]), language="text")

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
        with_logs=True,
        on_queue_update=on_queue_update,
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
    st.session_state.setdefault("link_results", [])


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def render_header() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <h1>Agentic Video Generator</h1>
            <p>Build the project config, gather source material, run the pipeline, review generated images, and compose the final video from one guided workspace.</p>
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
            if st.sidebar.button("Create / open project", use_container_width=True):
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

            submitted = st.form_submit_button("Save basics and continue", use_container_width=True)
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
            submitted = st.form_submit_button("Save outline and continue", use_container_width=True)
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
        if st.button("Save uploaded files", use_container_width=True):
            saved = save_uploaded_sources(project, uploaded)
            material = list(dict.fromkeys(config.get("source_material", []) + saved))
            config["source_material"] = material
            config["intro_material"] = list(dict.fromkeys(config.get("intro_material", []) + saved))
            save_config(project, config)
            st.success(f"Saved {len(saved)} uploaded source files.")

        link_text = st.text_area("Reference links or downloadable PDF/TXT links", value="\n".join(config.get("reference_links", [])), height=180)
        if st.button("Process links and save", use_container_width=True):
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

        if st.button("Save source links and continue", use_container_width=True):
            config["reference_links"] = list(dict.fromkeys(split_lines(link_text)))
            save_config(project, config)
            st.session_state.config_step = min(len(steps) - 1, st.session_state.config_step + 1)
            rerun_app()

    elif step == 3:
        tts_config = config.get("_project_config", {}).get("tts_config", {})
        validator_config = config.get("_project_config", {}).get("validator_config", {})
        with st.form("voice_form"):
            narration_style = st.text_area(
                "Narration style rules",
                value="\n".join(config.get("narration_style", [])),
                height=190,
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
            submitted = st.form_submit_button("Save voice settings and continue", use_container_width=True)
            if submitted:
                config["narration_style"] = split_lines(narration_style)
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
            submitted = st.form_submit_button("Save visual settings and continue", use_container_width=True)
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
            st.success("Saved config looks complete enough to run the pipeline.")

        st.json(config)
        st.caption("This page displays what is already saved on disk. It no longer overwrites config.json from stale defaults.")
        if st.button("Reload from disk", use_container_width=True):
            rerun_app()

    step_controls(len(steps))


def render_pipeline(project: str) -> None:
    st.subheader("Pipeline Runner")
    config = load_config(project)

    st.markdown("Run these first to prepare image review.")
    for label, script in PIPELINE_STAGES:
        st.markdown(f"<div class='stage-row'><strong>{label}</strong><br>{script}</div>", unsafe_allow_html=True)
        if st.button(f"Run {label}", key=f"run_{script}", use_container_width=True):
            run_stage(project, label, script)

    st.divider()
    st.markdown("After all images are approved, finish the video.")
    for label, script in FINISH_STAGES:
        actual_script = script or tts_script_for_config(config)
        st.markdown(f"<div class='stage-row'><strong>{label}</strong><br>{actual_script}</div>", unsafe_allow_html=True)
        if st.button(f"Run {label}", key=f"run_{label}", use_container_width=True):
            run_stage(project, label, actual_script)


def render_image_review(project: str) -> None:
    st.subheader("Image Review")
    config = load_config(project)
    items = load_image_prompt_items(project)
    if not items:
        st.info("Run Generate Image Prompts first. The app will then show each prompt here.")
        return

    st.session_state.active_review_index = min(st.session_state.active_review_index, len(items) - 1)
    item = items[st.session_state.active_review_index]
    section_index = item["section_index"]
    prompt_index = item["prompt_index"]

    approved_path = output_dir(project, "images") / f"image_{section_index}_{prompt_index}.png"
    review_dir = output_dir(project, "images") / "_review"
    rejected_dir = output_dir(project, "rejected_images")

    left, right = st.columns([2, 1])
    with left:
        st.caption(f"{st.session_state.active_review_index + 1} of {len(items)}")
        st.markdown(f"**Section {section_index}.{prompt_index}: {item['section_title']}**")
    with right:
        st.progress((st.session_state.active_review_index + 1) / len(items))

    if st.session_state.active_prompt_text == "":
        st.session_state.active_prompt_text = item["prompt"]

    st.text_area("Narration segment", value=item["narration"], height=120, disabled=True)
    prompt_text = st.text_area("Image prompt", value=st.session_state.active_prompt_text, height=180)
    st.session_state.active_prompt_text = prompt_text

    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        generate = st.button("Generate / regenerate", use_container_width=True)
    with cols[1]:
        keep = st.button("Keep image", disabled=not st.session_state.active_review_path, use_container_width=True)
    with cols[2]:
        reject = st.button("Reject image", disabled=not st.session_state.active_review_path, use_container_width=True)
    with cols[3]:
        skip = st.button("Next prompt", use_container_width=True)

    if generate:
        try:
            with st.spinner("Generating image..."):
                image = generate_image_with_fal(prompt_text, config)
                attempt = int(time.time())
                review_path = review_dir / f"image_{section_index}_{prompt_index}_attempt{attempt}.png"
                save_image_checked(image, review_path)
                st.session_state.active_review_path = str(review_path)
            st.success(f"Review image saved: {review_path}")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.active_review_path and Path(st.session_state.active_review_path).exists():
        st.image(st.session_state.active_review_path, caption="Current review image", use_container_width=True)
    elif approved_path.exists():
        st.image(str(approved_path), caption="Approved image already exists", use_container_width=True)

    if keep and st.session_state.active_review_path:
        try:
            copy_checked(Path(st.session_state.active_review_path), approved_path)
            st.success(f"Approved image saved: {approved_path}")
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
            st.warning(f"Rejected image saved: {reject_path}")
            st.session_state.active_review_path = ""
        except Exception as exc:
            st.error(str(exc))

    if skip:
        st.session_state.active_review_path = ""
        st.session_state.active_prompt_text = ""
        st.session_state.active_review_index = min(len(items) - 1, st.session_state.active_review_index + 1)
        rerun_app()

    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        if st.button("Previous prompt", disabled=st.session_state.active_review_index == 0):
            st.session_state.active_review_path = ""
            st.session_state.active_prompt_text = ""
            st.session_state.active_review_index -= 1
            rerun_app()
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
        ["Config Wizard", "Pipeline", "Image Review", "Outputs"],
        index=0,
    )

    st.markdown(f"Project: **{project}**")
    st.markdown(f"<div class='path-box'>{project_root(project)}</div>", unsafe_allow_html=True)

    if page == "Config Wizard":
        render_config_wizard(project)
    elif page == "Pipeline":
        render_pipeline(project)
    elif page == "Image Review":
        render_image_review(project)
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
