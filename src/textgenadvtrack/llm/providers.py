from __future__ import annotations

import os
from dataclasses import dataclass

import requests


class LLMProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model_name: str
    api_base: str | None = None
    api_key_env: str | None = None


def provider_config(provider: str, model_name: str) -> ProviderConfig:
    normalized = provider.lower()
    if normalized == "mock":
        return ProviderConfig(provider="mock", model_name=model_name)
    if normalized == "openai":
        return ProviderConfig(
            provider="openai",
            model_name=model_name,
            api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key_env="OPENAI_API_KEY",
        )
    raise ValueError(f"Unsupported provider: {provider}")


def generate_completion(
    provider: str,
    model_name: str,
    prompt: str,
    system_prompt: str | None = None,
    timeout: int = 120,
) -> str:
    config = provider_config(provider, model_name)
    if config.provider == "mock":
        prefix = f"[{model_name}] "
        if system_prompt:
            return f"{prefix}{system_prompt} | {prompt}"
        return f"{prefix}{prompt}"

    if config.provider == "openai":
        api_key = os.getenv(config.api_key_env or "")
        if not api_key:
            raise LLMProviderError(f"Missing API key env var: {config.api_key_env}")
        response = requests.post(
            f"{config.api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.model_name,
                "messages": [
                    *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("Unexpected provider response shape") from exc

    raise ValueError(f"Unhandled provider: {provider}")
