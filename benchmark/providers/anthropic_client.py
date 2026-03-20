from __future__ import annotations
from anthropic import Anthropic
from ..model_client import GenerationResult, Model


def _extract_text_from_content_blocks(content) -> str:
    """
    Anthropic returns content as a list of blocks;
    for a basic benchmark runner, we concatenate only text blocks.
    """
    if not content:
        return ""

    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "")
            if text:
                parts.append(text)

    return "".join(parts).strip()


def generate_anthropic(
    *,
    prompt: str,
    model: Model,
    system_prompt: str | None = None,
) -> GenerationResult:
    client = Anthropic()

    # gather request kwargs from the Model instance
    request_kwargs: dict = {
        "model": model.model_name,
        "max_tokens": model.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": prompt.strip(),
            }
        ],
    }

    if system_prompt and system_prompt.strip():
        request_kwargs["system"] = system_prompt.strip()

    if model.temperature is not None:
        request_kwargs["temperature"] = model.temperature

    if model.top_p is not None:
        request_kwargs["top_p"] = model.top_p

    if "top_k" in model.extra:
        request_kwargs["top_k"] = model.extra["top_k"]

    response = client.messages.create(**request_kwargs)

    # run the model
    usage = getattr(response, "usage", None)

    return GenerationResult(
        text=_extract_text_from_content_blocks(getattr(response, "content", None)),
        provider="anthropic",
        model_name=model.model_name,
        finish_reason=getattr(response, "stop_reason", None),
        usage_prompt_tokens=getattr(usage, "input_tokens", None) if usage else None,
        usage_completion_tokens=getattr(usage, "output_tokens", None) if usage else None,
        raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
    )