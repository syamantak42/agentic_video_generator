"""generate_images.py

Module for generating and curating images from prompts using FAL AI API.

Description:
    Interactive image generation and quality control interface. Reads image prompts from JSON,
    generates images using FAL AI's ByteDance SeedDream model, and allows interactive approval/
    rejection with optional prompt modification. Generates high-resolution images (2048x1152)
    with manual review pipeline.

Inputs:
    - outputs/output_jsons/image_prompts.json: Project structure with image_prompts list
    - source_material/config.json: Project metadata

Outputs:
    - outputs/images/image_*.png: Approved images
    - outputs/rejected_images/image_*_v*.png: Rejected images for review
    - outputs/images/_review/image_*_attempt*.png: Temporary review images

Environment Variables:
    - FAL_KEY: API key for FAL AI service

Usage:
    python generate_images.py <project_name> [section_start] [segment_start]
    
    Arguments:
        project_name: Project identifier (e.g., 'VikramBetaal')
        section_start: (Optional) Skip to section number (default: 1)
        segment_start: (Optional) Skip to segment within section (default: 1)
"""

import os
import sys
import json
import shutil
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

import fal_client
import requests

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

# -----------------------------
# Load environment & args
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(SCRIPT_DIR)
ENV_PATH = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=ENV_PATH)

project = sys.argv[1] if len(sys.argv) > 1 else None
project, source_dir, config = load_project_config(project)
project_root = os.path.dirname(source_dir)

# image model config (default: seedream-v4)
image_cfg = config.get("_project_config", {}).get("image_config", {})
image_model_key = image_cfg.get("model", "seedream-v4").strip().lower()
IMAGE_MODEL_MAP = {
    "seedream-v4": "fal-ai/bytedance/seedream/v4/text-to-image",
    "seedream_v4": "fal-ai/bytedance/seedream/v4/text-to-image",
}
IMAGE_MODEL = IMAGE_MODEL_MAP.get(image_model_key, image_model_key)
print(f"[IMAGE MODEL] {IMAGE_MODEL}")

# optional args
if len(sys.argv) >= 3:
    section_start = int(sys.argv[2])
else:
    section_start = 1

if len(sys.argv) >= 4:
    segment_start = int(sys.argv[3])
else:
    segment_start = 1

json_path = os.path.join(project_root, "outputs", "output_jsons", "image_prompts.json")
approved_dir = os.path.join(project_root, "outputs", "images")
rejected_dir = os.path.join(project_root, "outputs", "rejected_images")
review_dir = os.path.join(approved_dir, "_review")

historical_context = config.get("historical_context", "")
aesthetic_style = config.get("aesthetic_style", "")

os.makedirs(approved_dir, exist_ok=True)
os.makedirs(rejected_dir, exist_ok=True)
os.makedirs(review_dir, exist_ok=True)

print(f"[PROJECT] {project}")
print(f"[PROMPTS JSON] {json_path}")
print(f"[APPROVED IMAGES] {approved_dir}")
print(f"[REJECTED IMAGES] {rejected_dir}")
print(f"[REVIEW IMAGES] {review_dir}")

# -----------------------------
# Load JSON prompts
# -----------------------------
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

sections = data["sections"]
total_prompts = sum(len(section.get("image_prompts", [])) for section in sections)
if total_prompts == 0:
    raise RuntimeError(f"No image prompts found in {json_path}")
print(f"[PROMPTS FOUND] {total_prompts}")

# -----------------------------
# FAL key
# -----------------------------
FAL_KEY = os.getenv("FAL_KEY")
if not FAL_KEY:
    raise ValueError("Missing FAL_KEY in .env")

os.environ["FAL_KEY"] = FAL_KEY

# -----------------------------
# Queue callback
# -----------------------------
def on_queue_update(update):
    """
    Callback handler for FAL API queue updates during image generation.
    
    Args:
        update: Queue update object from FAL client
    
    Logs progress messages to console during generation.
    """    
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

# -----------------------------
# Image generation with FAL params
# -----------------------------
def generate_image(prompt):
    """
    Generate a single high-resolution image from text prompt using FAL AI.
    
    Args:
        prompt: Text description of desired image
    
    Returns:
        PIL.Image: RGB image (2048x1152 pixels)
    
    Raises:
        ValueError: If API returns no images
    
    Notes:
        - Uses ByteDance SeedDream v4 model
        - Safety checker disabled for creative content
        - Prompt enhancement set to standard mode
    """
    result = fal_client.subscribe(
        IMAGE_MODEL,
        arguments={
            "prompt": prompt,
            "image_size": {
                "height": 1152,
                "width": 2048
            },
            "num_images": 1,
            "max_images": 1,
            "enable_safety_checker": False,
            "enhance_prompt_mode": "standard"
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    if "images" not in result or len(result["images"]) == 0:
        raise ValueError("API returned no images")

    img_url = result["images"][0]["url"]
    resp = requests.get(img_url, timeout=60)
    resp.raise_for_status()
    img_bytes = resp.content
    return Image.open(BytesIO(img_bytes)).convert("RGB")

# -----------------------------
# INPUT NORMALIZATION
# -----------------------------
def get_decision(prompt_text):
    """
    Get user approval/rejection decision with flexible input handling.
    
    Args:
        prompt_text: Question prompt to display to user
    
    Returns:
        str: 'yes' or 'no' based on user input
    
    Input mapping:
        - ENTER, Y, or 'yes' => 'yes'
        - N or 'no' => 'no'
        - Other input => prompt retry
    """

    while True:
        try:
            raw = input(prompt_text)
        except EOFError as e:
            raise RuntimeError(
                "This script needs an interactive terminal so you can approve or reject images. "
                "Run it from PowerShell/CMD/VS Code Terminal, not a non-interactive runner."
            ) from e

        cleaned = raw.strip().lower()

        if cleaned == "":
            return "yes"
        if cleaned == "y":
            return "yes"
        if cleaned == "n":
            return "no"
        if cleaned == "yes":
            return "yes"
        if cleaned == "no":
            return "no"

        if cleaned in {"q", "quit", "exit"}:
            return "exit"

        print("Invalid input. Press ENTER/Y for yes, N for no, or Q to quit.")


def save_image_checked(img, path, label):
    img.save(path)
    if not os.path.exists(path):
        raise RuntimeError(f"{label} was not saved: {path}")

    size = os.path.getsize(path)
    if size <= 0:
        raise RuntimeError(f"{label} was saved as an empty file: {path}")

    print(f"{label} saved: {path} ({size} bytes)", flush=True)


def copy_image_checked(src_path, dst_path, label):
    if not os.path.exists(src_path):
        raise RuntimeError(f"Review image is missing, cannot save {label}: {src_path}")

    shutil.copy2(src_path, dst_path)
    if not os.path.exists(dst_path):
        raise RuntimeError(f"{label} was not saved: {dst_path}")

    size = os.path.getsize(dst_path)
    if size <= 0:
        raise RuntimeError(f"{label} was saved as an empty file: {dst_path}")

    print(f"{label} saved: {dst_path} ({size} bytes)", flush=True)


def show_image_for_review(img, review_path):
    """Save a generated image for review and try to open it without blocking approval."""
    save_image_checked(img, review_path, "Review image")

    try:
        img.show()
    except Exception as e:
        print(f"Could not open image viewer automatically: {e}")
        print("Open the review image path above, then choose whether to keep it.")

# -----------------------------
# MAIN LOOP (supports skipping)
# -----------------------------
generated_count = 0
approved_count = 0
rejected_count = 0

for section_index, section in enumerate(sections, start=1):

    if section_index < section_start:
        continue  # skip earlier sections

    prompts = section.get("image_prompts", [])
    narration_list = section.get("narration_text", [])

    print(f"\n===== SECTION {section_index}: {section['section_title']} =====\n")

    for prompt_index, original_prompt in enumerate(prompts, start=1):

        if section_index == section_start and prompt_index < segment_start:
            continue  # skip earlier prompts in same section

        narration_txt = narration_list[prompt_index - 1] if prompt_index - 1 < len(narration_list) else ""

        print("\n------------------------------------------------------------")
        print(f"NARRATION ({section_index}.{prompt_index}):\n{narration_txt}")
        print("------------------------------------------------------------")
        print(f"IMAGE PROMPT:\n{original_prompt}")
        print("------------------------------------------------------------\n")

        prompt = original_prompt

        # ------------------------------------------------------------
        # Confirm or modify prompt BEFORE FIRST GENERATION
        # ------------------------------------------------------------
        confirm = get_decision(
            "Confirm this prompt before generating? (ENTER/Y = yes, N = modify, Q = quit): "
        )
        if confirm == "exit":
            print("Exiting.")
            sys.exit()

        if confirm == "no":
            prompt = input("Enter new prompt: ").strip()
            print(f"Updated prompt:\n{prompt}\n")

        # ------------------------------------------------------------

        rejection_count = 0

        while True:
            attempt = rejection_count + 1
            print(f"\nGenerating Section {section_index}, Image {prompt_index} (attempt {attempt})")

            try:
                img = generate_image(prompt)
            except Exception as e:
                print(f"Error generating image: {e}")

                retry = get_decision("Retry? (ENTER/Y = yes, N = no, Q = quit): ")
                if retry == "exit":
                    print("Exiting.")
                    sys.exit()
                if retry == "yes":
                    continue
                else:
                    break

            review_filename = f"image_{section_index}_{prompt_index}_attempt{attempt}.png"
            review_path = os.path.join(review_dir, review_filename)
            show_image_for_review(img, review_path)
            generated_count += 1

            decision = get_decision("Keep this image? (ENTER/Y = keep, N = reject, Q = quit): ")

            if decision == "yes":
                filename = f"image_{section_index}_{prompt_index}.png"
                save_path = os.path.join(approved_dir, filename)
                copy_image_checked(review_path, save_path, "Approved image")
                approved_count += 1
                break

            elif decision == "no":
                rejection_count += 1
                rejected_count += 1
                filename = f"image_{section_index}_{prompt_index}_v{rejection_count}.png"
                save_path = os.path.join(rejected_dir, filename)
                copy_image_checked(review_path, save_path, "Rejected image")

                edit = get_decision("Modify prompt? (ENTER/Y = yes, N = no, Q = quit): ")
                if edit == "exit":
                    print("Exiting.")
                    sys.exit()
                if edit == "yes":
                    prompt = input("Enter new prompt: ").strip()
                continue

            elif decision == "exit":
                print("Exiting.")
                sys.exit()

if generated_count == 0:
    raise RuntimeError(
        "No images were generated. Check section_start/segment_start arguments and image_prompts.json."
    )

print(
    f"[DONE] generated={generated_count}, approved={approved_count}, rejected={rejected_count}",
    flush=True,
)

