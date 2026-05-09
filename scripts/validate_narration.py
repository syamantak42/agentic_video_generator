"""validate_narration.py

DeepSeek validator for generated narration scripts.

Inputs:
    - outputs/output_jsons/narration.json
    - outputs/output_jsons/outline_texts.json, if present
    - source_material/config.json

Output:
    - outputs/output_jsons/narration.json, rewritten with the same JSON structure
      after validation and editing
    - outputs/output_jsons/narration.before_narration_validation_*.json backup

Usage:
    python validate_narration.py <project_name>
"""

import json
import os

from validator_utils import load_project_config, read_json, validate_json_with_deepseek


def build_prompts(config, narration_data, outline_data):
    video_title = config.get("video_title", narration_data.get("video_title", ""))
    narration_style = config.get("narration_style", [])
    historical_context = config.get("historical_context", "")
    characters = config.get("characters", {})
    n_section = config.get("n_section", narration_data.get("n_sections", ""))

    system_prompt = """
You are a senior YouTube narration editor and validator.

Your job is to review a full generated narration JSON, silently judge whether it is a
good YouTube narration script, then return an improved JSON version.

Strict rules:
- Return ONLY valid JSON.
- Preserve the original JSON structure exactly.
- Preserve the same top-level type.
- Preserve every existing key exactly.
- Preserve the same number and order of sections.
- Preserve the same number and order of narration_text segments in every section.
- Do not add keys.
- Do not remove keys.
- Do not change lists into objects or objects into lists.
- Do not change non-text primitive types.
- Keep metadata such as video_title and n_sections unchanged unless the original
  structure already requires a string edit and the correction is clearly necessary.
- Edit textual narration and section title values where improvement is needed.

Editorial goals:
- Make the narration coherent as one complete video, not isolated paragraphs.
- Strengthen continuity and transitions between sections.
- Remove repetition, filler, vague phrasing, and generic LLM-sounding language.
- Maintain the user's requested tone and content constraints.
- Preserve factual caution and avoid introducing unsupported claims.
- Keep each segment natural for spoken narration.
""".strip()

    user_prompt = f"""
PROJECT CONTEXT
Video title:
{video_title}

Expected section count:
{n_section}

Narration style requirements:
{json.dumps(narration_style, ensure_ascii=False, indent=2)}

Historical / subject context:
{historical_context}

Character canon:
{json.dumps(characters, ensure_ascii=False, indent=2)}

VALIDATED OUTLINE JSON, if available:
{json.dumps(outline_data, ensure_ascii=False, indent=2)}

TASK
Review the narration JSON below as a validator agent.
Ask yourself internally:
- Does this work as a coherent YouTube video narration from opening to ending?
- Does it satisfy the outline and the user's tone requirements?
- Are the transitions smooth enough?
- Are there contradictions, repetitions, weak passages, filler, or generic phrasing?
- Is each narration_text segment clear, spoken, focused, and useful?
- Does the video maintain audience interest without becoming shallow or clickbait?

Then edit the narration JSON to make it more coherent, polished, and task-aligned.

Return ONLY the revised JSON. Keep the original JSON structure exactly.

ORIGINAL NARRATION JSON
{json.dumps(narration_data, ensure_ascii=False, indent=2)}
""".strip()

    return system_prompt, user_prompt


def main():
    project, source_dir, config = load_project_config()
    project_root = os.path.dirname(source_dir)
    json_dir = os.path.join(project_root, "outputs", "output_jsons")
    narration_path = os.path.join(json_dir, "narration.json")
    outline_path = os.path.join(json_dir, "outline_texts.json")

    if not os.path.exists(narration_path):
        raise FileNotFoundError(f"Missing narration JSON: {narration_path}")

    narration_data = read_json(narration_path)
    outline_data = read_json(outline_path) if os.path.exists(outline_path) else {}

    system_prompt, user_prompt = build_prompts(config, narration_data, outline_data)
    result = validate_json_with_deepseek(
        json_path=narration_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
        label="narration_validation",
    )

    print(f"[OK] Narration validated with {result['model']}")
    print(f"[OK] Backup saved to {result['backup_path']}")
    print(f"[OK] Updated JSON saved to {result['json_path']}")


if __name__ == "__main__":
    main()
