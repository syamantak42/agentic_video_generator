"""generate_image_prompts.py

Module for generating visual image prompts for video sections using LLM.

Description:
    This script reads narration text and project metadata, then uses DeepSeek LLM to
    generate detailed, visually descriptive image prompts for each narration segment.
    To minimize visual repetition, it maintains summaries of previous prompts and
    instructs the LLM to create varied compositions while keeping character identity,
    historical context, and aesthetic style consistent.

Inputs:
    - outputs/output_jsons/narration.json: Structured narration with sections, titles, and narration_text segments
    - source_material/config.json: Project metadata (historical_context, characters, aesthetic_style)

Outputs:
    - outputs/output_jsons/image_prompts.json: Full project structure augmented with:
        * image_prompts: List of concise visual descriptions (one per narration segment)
        * image_prompt_summaries: Short semantic summaries for diversity control

Environment Variables:
    - DEEPSEEK_API_KEY: API key for DeepSeek language model

Usage:
    python generate_image_prompts.py <project_name>
"""

import os
import sys
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from console_utils import configure_utf8_output
from prompt_loader import render_prompt
from text_utils import clean_json_text, clean_text

configure_utf8_output()

# --------------------------------------------------------
# Helper: Get project name and paths from config
# --------------------------------------------------------
def load_project_config(project_arg=None):
    """Load project name from config.json or command-line arg."""
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

# ----------------------------- Env -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in .env file")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# --------------------------- Paths -----------------------------
project, source_dir, config = load_project_config()
# base directories relative to this script
project_root = os.path.dirname(source_dir)
SCRIPTS_DIR = os.path.join(project_root, "outputs", "output_jsons")
NARRATION_PATH = os.path.join(SCRIPTS_DIR, "narration.json")
OUTPUT_PATH = os.path.join(SCRIPTS_DIR, "image_prompts.json")

if not os.path.exists(NARRATION_PATH):
    raise FileNotFoundError(f"Missing narration JSON: {NARRATION_PATH}")

# ---------------------- Load inputs ----------------------------
with open(NARRATION_PATH, "r", encoding="utf-8") as f:
    narration_data = json.load(f)
video_title = narration_data["video_title"]
sections = narration_data["sections"]

historical_context = config.get("historical_context", "")
characters = config.get("characters", {})
aesthetic_style = config.get("aesthetic_style", "")

print(f"[LOAD] {len(sections)} sections | title: {video_title}")
print(f"[CTX] historical_context: {bool(historical_context)} | characters: {len(characters)} | style: {bool(aesthetic_style)}")

# ---------------------- LLM helpers ----------------------------
def summarize_single_prompt(client_, prompt_text: str) -> str:
    """
    Summarize an image prompt into a concise semantic descriptor.
    
    Args:
        client_: OpenAI client instance for DeepSeek API
        prompt_text: Full image prompt text to summarize
    
    Returns:
        str: Single-sentence summary capturing visual scene, setting, and main action.
             Excludes artistic style but preserves camera angle if present.
    """
    prompt = render_prompt(
        "generate_image_prompts_summary_user.txt",
        prompt_text=prompt_text,
    )
    resp = client_.chat.completions.create(
        model="deepseek-chat",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt.strip()}],
    )
    return clean_text(resp.choices[0].message.content.strip())

def generate_image_prompt(client_, narration_text: str, prior_summaries: list[str]) -> str:
    """
    Generate a detailed image prompt for a narration segment.
    
    Args:
        client_: OpenAI client instance for DeepSeek API
        narration_text: The narration text to create a visual prompt for
        prior_summaries: List of semantic summaries from previously generated prompts (typically last 8)
    
    Returns:
        str: Concise, detailed image prompt (<150 words) with explicit character descriptions,
             historical context, and aesthetic style. No maps, violence, or sexual content.
    
    Notes:
        - Enforces visual diversity by considering prior_summaries
        - Includes specific character details from canon
        - Appends historical context and aesthetic style to final prompt
    """
    # Keep context compact: only the last ~8 summaries
    recent_summaries = prior_summaries[-8:] if prior_summaries else []
    summaries_block = "\n".join(f"- {s}" for s in recent_summaries) if recent_summaries else "None yet."
    #print("summaries: \n" + summaries_block)

    characters_json = json.dumps(characters, ensure_ascii=False, indent=2)
    sys_prompt = render_prompt(
        "generate_image_prompts_system.txt",
        video_title=video_title,
        historical_context=historical_context,
        characters_json=characters_json,
    )

    user_prompt = render_prompt(
        "generate_image_prompts_user.txt",
        narration_text=narration_text,
        characters_json=characters_json,
        summaries_block=summaries_block,
    )

    resp = client_.chat.completions.create(
        model="deepseek-chat",
        temperature=1.3,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = clean_text(resp.choices[0].message.content.strip())
    content = re.sub(r"^```.*|```$", "", content, flags=re.MULTILINE).strip()
    image_prompt = f"{content}, set in {historical_context}, rendered in {aesthetic_style}"
    return image_prompt

# ------------------------- Main loop ---------------------------
all_prompt_summaries: list[str] = []
total = sum(len(s["narration_text"]) for s in sections)
count = 0

for sec_idx, sec in enumerate(sections, start=1):
    title = sec["section_title"]
    texts = sec["narration_text"]

    sec_image_prompts = []
    sec_image_summaries = []

    print(f"\n[Section {sec_idx}] {title}")
    for t in texts:
        count += 1
        print(f"  Generating image prompt {count}/{total} ...")

        img_prompt = generate_image_prompt(client, t, all_prompt_summaries)
        print(img_prompt)
        sec_image_prompts.append(img_prompt)

        summary = summarize_single_prompt(client, img_prompt)
        sec_image_summaries.append(summary)
        all_prompt_summaries.append(summary)

    sec["image_prompts"] = sec_image_prompts
    sec["image_prompt_summaries"] = sec_image_summaries

# ------------------------ Save output -------------------------
out_data = {
    "video_title": video_title,
    "n_sections": len(sections),
    "historical_context": historical_context,
    "aesthetic_style": aesthetic_style,
    "characters": characters,
    "sections": sections,
}

out_data = clean_json_text(out_data)
os.makedirs(SCRIPTS_DIR, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out_data, f, ensure_ascii=False, indent=2)

print(f"\n[OK] Image prompts saved to {OUTPUT_PATH}")
