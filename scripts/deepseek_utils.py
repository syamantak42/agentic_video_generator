"""Shared DeepSeek configuration helpers."""

import os


DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_MODEL_CHOICES = ["deepseek-v4-flash", "deepseek-v4-pro"]

DEEPSEEK_COMPLETION_KWARGS = {
    "reasoning_effort": "high",
    "extra_body": {"thinking": {"type": "enabled"}},
}


def get_deepseek_model(config):
    project_config = config.get("_project_config", {})
    validator_config = project_config.get("validator_config", {})
    configured_model = validator_config.get("model")
    if configured_model in DEEPSEEK_MODEL_CHOICES:
        return configured_model

    env_model = os.getenv("DEEPSEEK_MODEL")
    if env_model in DEEPSEEK_MODEL_CHOICES:
        return env_model

    return DEFAULT_DEEPSEEK_MODEL


def create_deepseek_chat_completion(client, **kwargs):
    request = dict(DEEPSEEK_COMPLETION_KWARGS)
    request.update(kwargs)
    return client.chat.completions.create(**request)
