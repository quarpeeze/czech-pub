from __future__ import annotations

from google import genai
from google.genai import types

from ..model_client import GenerationResult, Model


def generate_google(
    *,
    prompt: str,
    model: Model,
    system_prompt: str | None = None,
) -> GenerationResult:
    client = genai.Client()

    config_kwargs: dict = {}

    if system_prompt and system_prompt.strip():
        config_kwargs["system_instruction"] = system_prompt.strip()

    if model.max_tokens is not None:
        config_kwargs["max_output_tokens"] = model.max_tokens

    if model.temperature is not None:
        config_kwargs["temperature"] = model.temperature

    if model.top_p is not None:
        config_kwargs["top_p"] = model.top_p

    if "top_k" in model.extra:
        config_kwargs["top_k"] = model.extra["top_k"]

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    response = client.models.generate_content(
        model=model.model_name,
        contents=prompt.strip(),
        config=config,
    )

    usage = getattr(response, "usage_metadata", None)

    return GenerationResult(
        text=getattr(response, "text", "") or "",
        provider="google",
        model_name=model.model_name,
        finish_reason=str(getattr(response, "finish_reason", None)) if getattr(response, "finish_reason", None) is not None else None,
        usage_prompt_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
        usage_completion_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
        raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
    )