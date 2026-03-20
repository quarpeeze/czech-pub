from __future__ import annotations

from openai import OpenAI

from ..entities import GenerationResult, Model


def _build_input_text(prompt: str, system_prompt: str | None) -> str:
    """
    prompt is neccessary, system prompt currently is not
    """
    if system_prompt and system_prompt.strip():
        return f"{system_prompt.strip()}\n\n{prompt.strip()}"
    return prompt.strip()


def generate_openai(
    *,
    prompt: str,
    model: Model,
    system_prompt: str | None = None,
) -> GenerationResult:
    client = OpenAI()

    input_text = _build_input_text(prompt, system_prompt) # do we use system prompt?

    # gather request kwargs from the Model instance
    request_kwargs: dict = {
        "model": model.model_name,
        "input": input_text,
    }
    
    if model.max_tokens is not None:
        request_kwargs["max_output_tokens"] = model.max_tokens

    if model.temperature is not None:
        request_kwargs["temperature"] = model.temperature

    if model.top_p is not None:
        request_kwargs["top_p"] = model.top_p

    response = client.responses.create(**request_kwargs)

    usage = getattr(response, "usage", None)
    output = getattr(response, "output", None)

    finish_reason = None
    if output and isinstance(output, list):
        first = output[0]
        finish_reason = getattr(first, "status", None)

    prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
    completion_tokens = getattr(usage, "output_tokens", None) if usage else None

    return GenerationResult(
        text=response.output_text or "",
        provider="openai",
        model_name=model.model_name,
        finish_reason=finish_reason,
        usage_prompt_tokens=prompt_tokens,
        usage_completion_tokens=completion_tokens,
        raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
    )