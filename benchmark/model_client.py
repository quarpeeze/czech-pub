from __future__ import annotations

from .entities import Model, GenerationResult


def _raise_missing_dependency(provider: str, package_note: str, error: ImportError) -> None:
    raise ImportError(
        f"provider '{provider}' requires {package_note}. "
        f"Install the matching dependency set before using this provider."
    ) from error


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
        try:
            from .providers.openai_client import generate_openai
        except ImportError as e:
            _raise_missing_dependency("openai", "the `openai` package", e)
        return generate_openai(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        )

    if provider == "anthropic":
        try:
            from .providers.anthropic_client import generate_anthropic
        except ImportError as e:
            _raise_missing_dependency("anthropic", "the `anthropic` package", e)
        return generate_anthropic(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
    )

    if provider == "google":
        try:
            from .providers.google_client import generate_google
        except ImportError as e:
            _raise_missing_dependency("google", "the `google-genai` package", e)
        return generate_google(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        )

    if provider == "hf_local":
        try:
            from .providers.hf_local_client import generate_hf_local
        except ImportError as e:
            _raise_missing_dependency(
                "hf_local",
                "`transformers`, `accelerate`, and a local `torch` installation",
                e,
            )
        return generate_hf_local(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        )

    if provider == "random_baseline":
        from .providers.random_baseline import generate_random_baseline
        return generate_random_baseline(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        )

    if provider == "together":
        try:
            from .providers.together_client import generate_together
        except ImportError as e:
            _raise_missing_dependency("together", "the `together` package", e)
        return generate_together(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        )
    

    raise ValueError(f"unsupported provider: {model.provider}")
