from __future__ import annotations

import os
from together import Together

from ..entities import GenerationResult, Model


def generate_together(
    *,
    prompt: str,
    model: Model,
    system_prompt: str | None = None,
) -> GenerationResult:
    """
    run one generation call via Together chat completions
    """

    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY is not set in environment variables.")

    client = Together(api_key=api_key)

    messages: list[dict] = []

    if system_prompt and system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": system_prompt.strip(),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt.strip(),
        }
    )

    request_kwargs: dict = {
        "model": model.model_name,
        "messages": messages,
    }

    if "Qwen/Qwen3.5" in model.model_name:
        request_kwargs["reasoning"] = {"enabled": False}

    if model.model_name in {"deepseek-ai/DeepSeek-V3.1", "deepseek-v3.1"}:
        request_kwargs["reasoning"] = {"enabled": False}

    if model.max_tokens is not None:
        request_kwargs["max_tokens"] = model.max_tokens

    if model.temperature is not None:
        request_kwargs["temperature"] = model.temperature

    if model.top_p is not None:
        request_kwargs["top_p"] = model.top_p

    # allow optional extra params if needed later
    for key, value in model.extra.items():
        request_kwargs[key] = value

    response = client.chat.completions.create(**request_kwargs)

    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None) if choice else None
    usage = getattr(response, "usage", None)

    text = ""
    if message is not None:
        text = getattr(message, "content", "") or ""

    finish_reason = getattr(choice, "finish_reason", None) if choice else None

    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

    return GenerationResult(
        text=text.strip(),
        provider="together",
        model_name=model.model_name,
        finish_reason=finish_reason,
        usage_prompt_tokens=prompt_tokens,
        usage_completion_tokens=completion_tokens,
        raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
    )