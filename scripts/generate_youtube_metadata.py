"""Generate YouTube metadata from narration.json.

Inputs:
    - outputs/output_jsons/narration.json
    - source_material/config.json

Output:
    - outputs/output_jsons/youtube_metadata.json

Usage:
    python generate_youtube_metadata.py <project_name>
"""

from __future__ import annotations

import json
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

from console_utils import configure_utf8_output
from deepseek_utils import create_deepseek_chat_completion, get_deepseek_model
from prompt_loader import load_prompt, render_prompt
from text_utils import clean_json_text, clean_text


configure_utf8_output()


def load_project_config(project_arg: str | None = None) -> tuple[str, str, dict]:
    project = project_arg or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not project:
        raise ValueError("Project name required: python generate_youtube_metadata.py PROJECT_NAME")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.join(base_dir, project)
    source_dir = os.path.join(project_root, "source_material")
    config_path = os.path.join(source_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    project_name = config.get("_project_config", {}).get("project_name", project)
    return project_name, source_dir, config


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def narration_to_text(narration_data: dict) -> str:
    lines: list[str] = []
    sections = narration_data.get("sections", [])
    if not sections:
        raise ValueError("narration.json contains no sections.")

    for section_index, section in enumerate(sections, start=1):
        title = clean_text(section.get("section_title", f"Section {section_index}"))
        lines.append(f"SECTION {section_index}: {title}")
        narration_segments = section.get("narration_text", [])
        if not isinstance(narration_segments, list) or not narration_segments:
            raise ValueError(f"Section {section_index} has no narration_text segments.")
        for segment_index, segment in enumerate(narration_segments, start=1):
            lines.append(f"{section_index}.{segment_index}: {clean_text(str(segment))}")
        lines.append("")

    return "\n".join(lines).strip()


def parse_metadata_response(content: str) -> dict:
    cleaned = clean_text(content)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Metadata response was not a JSON object.")

    title = clean_text(data.get("title", ""))
    description = clean_text(data.get("description", ""))
    tags = data.get("tags", [])

    if not title:
        raise ValueError("Metadata response is missing title.")
    if not description:
        raise ValueError("Metadata response is missing description.")
    if not isinstance(tags, list):
        raise ValueError("Metadata response tags must be a list.")

    cleaned_tags = []
    for tag in tags:
        tag_text = clean_text(str(tag)).lstrip("#").strip()
        if tag_text and tag_text not in cleaned_tags:
            cleaned_tags.append(tag_text)

    if not cleaned_tags:
        raise ValueError("Metadata response contains no tags.")

    return clean_json_text(
        {
            "title": title[:100],
            "description": description,
            "tags": cleaned_tags,
        }
    )


def main() -> None:
    project, source_dir, config = load_project_config()
    project_root = os.path.dirname(source_dir)
    json_dir = os.path.join(project_root, "outputs", "output_jsons")
    narration_path = os.path.join(json_dir, "narration.json")
    output_path = os.path.join(json_dir, "youtube_metadata.json")

    if not os.path.exists(narration_path):
        raise FileNotFoundError(f"Missing narration JSON: {narration_path}")

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY in .env file")

    narration_data = read_json(narration_path)
    video_title = config.get("video_title", narration_data.get("video_title", project))
    narration_text = narration_to_text(narration_data)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = create_deepseek_chat_completion(
        client,
        model=get_deepseek_model(config),
        temperature=0.8,
        messages=[
            {"role": "system", "content": load_prompt("generate_youtube_metadata_system.txt")},
            {
                "role": "user",
                "content": render_prompt(
                    "generate_youtube_metadata_user.txt",
                    video_title=video_title,
                    narration_text=narration_text,
                ),
            },
        ],
    )

    metadata = parse_metadata_response(response.choices[0].message.content or "")
    os.makedirs(json_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] YouTube metadata saved to {output_path}")


if __name__ == "__main__":
    main()
