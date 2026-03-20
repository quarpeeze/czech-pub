from __future__ import annotations

from .entities import Model, GenerationResult

from .providers.openai_client import generate_openai
from .providers.anthropic_client import generate_anthropic
from .providers.google_client import generate_google
# from .providers.hf_local_client import generate_hf_local


def generate(
    *,
    prompt: str,
    model: Model,
    system_prompt: str | None = None,
) -> GenerationResult:
    """
    provider-agnostic generation entrypoint
    """

    provider = model.provider.lower().strip()

    if provider == "openai":
        return generate_openai(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        )

    if provider == "anthropic":
        return generate_anthropic(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
    )

    if provider == "google":
        return generate_google(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        )

    # if provider == "hf_local":
    #     return generate_hf_local(
    #         prompt=prompt,
    #         model=model,
    #         system_prompt=system_prompt,
    #     )

    raise ValueError(f"unsupported provider: {model.provider}")