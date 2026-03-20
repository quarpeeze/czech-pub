from __future__ import annotations

from functools import lru_cache
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..model_client import GenerationResult, Model


@lru_cache(maxsize=4)
def _load_tokenizer_and_model(model_name: str):
    """
    Loads tokenizer fot the given model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    return tokenizer, model


def _build_model_inputs(
    *,
    tokenizer,
    prompt: str,
    system_prompt: str | None,
):
    """
    Build tokenized inputs for the model.

    If the tokenizer has a chat template, use it.
    Otherwise, fall back to plain text concatenation.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = []

        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        messages.append({"role": "user", "content": prompt.strip()})

        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    # Fallback for non-chat models or models without a chat template
    full_prompt = prompt.strip()
    if system_prompt and system_prompt.strip():
        full_prompt = f"{system_prompt.strip()}\n\n{prompt.strip()}"

    return tokenizer(full_prompt, return_tensors="pt")


def generate_hf_local(
    *,
    prompt: str,
    model: Model,
    system_prompt: str | None = None,
) -> GenerationResult:
    tokenizer, hf_model = _load_tokenizer_and_model(model.model_name)

    inputs = _build_model_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        system_prompt=system_prompt,
    )

    # Move tokenized inputs to the model device when possible.
    # With device_map="auto", model may be sharded, but input tensors still need
    # an initial device; model.device is usually fine for the entry tensors.
    if hasattr(inputs, "to"):
        inputs = inputs.to(hf_model.device)
    else:
        inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": model.max_tokens,
        "do_sample": False if model.temperature in (None, 0, 0.0) else True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if model.temperature is not None:
        generate_kwargs["temperature"] = model.temperature

    if model.top_p is not None:
        generate_kwargs["top_p"] = model.top_p

    # Allow provider-specific extras later without polluting the common Model fields
    for key in ["top_k", "repetition_penalty"]:
        if key in model.extra:
            generate_kwargs[key] = model.extra[key]

    started = perf_counter()
    with torch.no_grad():
        output_ids = hf_model.generate(**inputs, **generate_kwargs)
    latency_sec = perf_counter() - started

    # Decode only newly generated tokens, not the original prompt
    input_len = inputs["input_ids"].shape[1]
    new_token_ids = output_ids[0][input_len:]
    text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    return GenerationResult(
        text=text,
        provider="hf_local",
        model_name=model.model_name,
        finish_reason=None,
        usage_prompt_tokens=int(input_len),
        usage_completion_tokens=int(new_token_ids.shape[0]),
        raw_response={
            "latency_sec": round(latency_sec, 4),
            "generated_token_count": int(new_token_ids.shape[0]),
        },
    )