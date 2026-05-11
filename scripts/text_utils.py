"""Text cleanup helpers for generated artifacts."""

import re


UNICODE_REPLACEMENTS = {
    "\u00a0": " ",
    "\u00ad": "",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
    "\u2060": "",
    "\ufeff": "",
    "\ufffd": "",
}


def clean_text(text):
    """Clean generated text without flattening meaningful punctuation."""
    if not isinstance(text, str):
        return text

    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    return text.strip()


def clean_json_text(value):
    """Recursively clean strings inside JSON-like structures."""
    if isinstance(value, dict):
        return {key: clean_json_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_text(item) for item in value]
    if isinstance(value, str):
        return clean_text(value)
    return value
