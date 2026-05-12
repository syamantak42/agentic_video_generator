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
from deepseek_utils import DEFAULT_DEEPSEEK_MODEL, create_deepseek_chat_completion, get_deepseek_model
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
MODEL_NAME = DEFAULT_DEEPSEEK_MODEL

# --------------------------- Paths -----------------------------
project, source_dir, config = load_project_config()
MODEL_NAME = get_deepseek_model(config)
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

# --------------------- Diversity controls ---------------------
SUMMARY_LOOKBACK = 10
SIMILARITY_RETRY_THRESHOLD = 0.48

TOKEN_STOPWORDS = {
    "about", "above", "across", "after", "again", "against", "also", "amid",
    "among", "and", "are", "around", "background", "before", "behind",
    "below", "between", "both", "camera", "clear", "close", "composition",
    "depicts", "during", "each", "from", "front", "image", "into", "main",
    "near", "only", "over", "prompt", "rendered", "scene", "setting",
    "shows", "still", "subject", "that", "the", "their", "there", "through",
    "under", "visible", "with", "within",
}

# ---------------------- LLM helpers ----------------------------
def build_characters_block() -> str:
    """Return character guidance without encouraging invented canon details."""
    if characters:
        return json.dumps(characters, ensure_ascii=False, indent=2)
    return (
        "No fixed character canon is provided. Do not invent named ethnic, cultural, "
        "caste, clan, or tribal groups. Describe people only by visible age range, "
        "clothing, posture, expression, and role in the scene."
    )


def build_prior_visuals(prior_summaries: list[str]) -> str:
    """Format prior prompt summaries as visual frames to avoid."""
    recent_summaries = prior_summaries[-SUMMARY_LOOKBACK:] if prior_summaries else []
    if not recent_summaries:
        return "None yet."
    return "\n".join(f"{idx}. {summary}" for idx, summary in enumerate(recent_summaries, start=1))


def clean_prompt_response(content: str) -> str:
    """Remove common chat artifacts while keeping the prompt as one clean line."""
    content = re.sub(r"```(?:\w+)?", "", content)
    content = content.replace("```", "")
    content = re.sub(
        r"(?im)^\s*(image prompt|prompt|response\s*[a-z]?)\s*[:\-]\s*",
        "",
        content,
    )
    content = re.sub(r"(?i)\bresponse\s*[a-z]?\s*[:\-]\s*", "", content)
    content = " ".join(content.split())
    return clean_text(content).strip(" \"'")


def append_context_and_style(prompt: str) -> str:
    """Append project context only when values are present."""
    suffixes = []
    context = str(historical_context).strip()
    style = str(aesthetic_style).strip()
    if context:
        suffixes.append(f"set in {context}")
    if style:
        suffixes.append(f"rendered in {style}")
    if suffixes:
        prompt = f"{prompt.rstrip('.,')}, {', '.join(suffixes)}"
    return clean_text(prompt)


def visual_tokens(text: str) -> set[str]:
    """Tokenize a visual fingerprint for lightweight similarity detection."""
    normalized = clean_text(text.lower())
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    return {token for token in tokens if len(token) > 2 and token not in TOKEN_STOPWORDS}


def strongest_visual_overlap(summary: str, prior_summaries: list[str]) -> tuple[float, str]:
    """Return the strongest Jaccard overlap against recent visual summaries."""
    current_tokens = visual_tokens(summary)
    if not current_tokens:
        return 0.0, ""

    best_score = 0.0
    best_match = ""
    for prior in prior_summaries[-SUMMARY_LOOKBACK:]:
        prior_tokens = visual_tokens(prior)
        if not prior_tokens:
            continue
        score = len(current_tokens & prior_tokens) / len(current_tokens | prior_tokens)
        if score > best_score:
            best_score = score
            best_match = prior
    return best_score, best_match


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
    resp = create_deepseek_chat_completion(
        client_,
        model=MODEL_NAME,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt.strip()}],
    )
    return clean_text(resp.choices[0].message.content.strip())

def generate_image_prompt(
    client_,
    narration_text: str,
    prior_summaries: list[str],
) -> str:
    """
    Generate a detailed image prompt for a narration segment.
    
    Args:
        client_: OpenAI client instance for DeepSeek API
        narration_text: The narration text to create a visual prompt for
        prior_summaries: Semantic summaries from previously generated prompts
    
    Returns:
        str: Concise, detailed image prompt (<150 words) with explicit character descriptions,
             historical context, and aesthetic style. No maps, violence, or sexual content.
    
    Notes:
        - Enforces visual diversity by considering prior_summaries
        - Includes specific character details from canon
        - Appends historical context and aesthetic style to final prompt
    """
    prior_visuals = build_prior_visuals(prior_summaries)
    characters_json = build_characters_block()

    sys_prompt = render_prompt(
        "generate_image_prompts_system.txt",
        video_title=video_title,
        historical_context=historical_context,
        characters_json=characters_json,
    )

    user_prompt = render_prompt(
        "generate_image_prompts_user.txt",
        narration_text=narration_text,
        prior_visuals=prior_visuals,
    )

    resp = create_deepseek_chat_completion(
        client_,
        model=MODEL_NAME,
        temperature=1.3,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = clean_prompt_response(resp.choices[0].message.content.strip())
    if not content:
        raise ValueError("Image prompt generation returned empty content.")
    return append_context_and_style(content)

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
        print(f"Generating image prompt {count}/{total}", flush=True)

        img_prompt = generate_image_prompt(client, t, all_prompt_summaries)
        summary = summarize_single_prompt(client, img_prompt)

        similarity_score, similar_prior = strongest_visual_overlap(summary, all_prompt_summaries)
        if similarity_score >= SIMILARITY_RETRY_THRESHOLD:
            print(
                "  [DIVERSITY] Near-duplicate visual frame detected "
                f"({similarity_score:.0%} overlap); regenerating once ..."
            )
            img_prompt = generate_image_prompt(
                client,
                t,
                all_prompt_summaries + [summary],
            )
            summary = summarize_single_prompt(client, img_prompt)

        print(img_prompt)
        sec_image_prompts.append(img_prompt)
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
