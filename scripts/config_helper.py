"""Small DeepSeek helpers for filling config wizard fields."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from deepseek_utils import DEFAULT_DEEPSEEK_MODEL, DEEPSEEK_COMPLETION_KWARGS
from dotenv import load_dotenv
from openai import OpenAI
from text_utils import clean_text


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENV_PATH = REPO_ROOT / ".env"

SYSTEM_PROMPT = (
    "You help fill a video generator config. "
    "Return only the requested field text. "
    "Do not include markdown, code fences, headings, or explanations."
)

GUIDELINES_MAX_TOKENS = 900
NARRATION_STYLE_MAX_TOKENS = 450
AESTHETIC_STYLE_MAX_TOKENS = 300


def get_client() -> OpenAI:
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY in .env file")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def ask_deepseek(user_prompt: str, max_tokens: int = 600) -> str:
    response = get_client().chat.completions.create(
        model=DEFAULT_DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=max_tokens,
        **DEEPSEEK_COMPLETION_KWARGS,
    )
    return clean_text(response.choices[0].message.content or "")


def generate_guidelines(video_title: str, n_section: int) -> str:
    prompt = (
        f"Generate a brief but concise and compact outline for a YouTube video on {video_title} "
        f"with {n_section} sections.\n"
        f"Return exactly {n_section} short lines, one line per section."
    )
    return ask_deepseek(prompt, max_tokens=GUIDELINES_MAX_TOKENS)


def generate_narration_style(video_title: str) -> str:
    prompt = (
        f"Suggest an appropriate narration style for a YouTube video on {video_title}.\n"
        "Return 3 to 5 short lines."
    )
    return ask_deepseek(prompt, max_tokens=NARRATION_STYLE_MAX_TOKENS)


def generate_aesthetic_style(video_title: str) -> str:
    prompt = (
        f"Suggest an aesthetic style for a video on {video_title}.\n"
        "Return one concise visual style sentence. "
        "The Aesthetic style MUST be coherent, and easy to render by standard image generation models."
    )
    return ask_deepseek(prompt, max_tokens=AESTHETIC_STYLE_MAX_TOKENS)


def generate_all(video_title: str, n_section: int) -> dict:
    return {
        "guidelines": generate_guidelines(video_title, n_section),
        "narration_style": generate_narration_style(video_title),
        "aesthetic_style": generate_aesthetic_style(video_title),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate config wizard suggestions with DeepSeek.")
    parser.add_argument(
        "--task",
        choices=["guidelines", "narration_style", "aesthetic_style", "all"],
        required=True,
    )
    parser.add_argument("--video-title", required=True)
    parser.add_argument("--n-section", type=int, default=6)
    args = parser.parse_args()

    if args.task == "guidelines":
        result = generate_guidelines(args.video_title, args.n_section)
    elif args.task == "narration_style":
        result = generate_narration_style(args.video_title)
    elif args.task == "aesthetic_style":
        result = generate_aesthetic_style(args.video_title)
    else:
        result = generate_all(args.video_title, args.n_section)

    print(json.dumps({"task": args.task, "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
