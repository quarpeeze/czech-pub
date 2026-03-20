from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Model:
    provider: str
    model_name: str
    max_tokens: int = 128
    temperature: float | None = None
    top_p: float | None = None
    timeout_sec: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    provider: str
    model_name: str
    finish_reason: str | None = None
    usage_prompt_tokens: int | None = None
    usage_completion_tokens: int | None = None
    raw_response: Any | None = None