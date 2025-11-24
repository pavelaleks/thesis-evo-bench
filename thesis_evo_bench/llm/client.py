"""LLM client wrapper for DeepSeek API interactions."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import httpx

from thesis_evo_bench.config import get_settings

logger = logging.getLogger(__name__)


async def call_llm(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    retries: int = 3,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """
    Call DeepSeek LLM API asynchronously.

    Args:
        model: Model name (defaults to deepseek-reasoner)
        prompt: User prompt text
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate (None = model default)
        system_prompt: Optional system prompt
        retries: Number of retry attempts on failure
        timeout: Request timeout in seconds

    Returns:
        Dictionary with 'content' (text response) and 'usage' (token usage)

    Raises:
        httpx.HTTPError: On HTTP errors after retries
        ValueError: On invalid response format
    """
    settings = get_settings()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    # Construct full URL: base_url should be "https://api.deepseek.com/v1"
    # and we append "/chat/completions"
    base_url = settings.llm_base_url.rstrip('/')
    if not base_url.endswith('/v1'):
        # Ensure base URL ends with /v1
        if base_url.endswith('/v1/chat/completions'):
            base_url = base_url.replace('/v1/chat/completions', '/v1')
        elif '/v1' not in base_url:
            base_url = f"{base_url}/v1"
    api_url = f"{base_url}/chat/completions"

    last_error = None
    for attempt in range(retries):
        try:
            logger.debug(f"Calling LLM API: {api_url} (attempt {attempt + 1}/{retries})")
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    api_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                if "choices" not in data or not data["choices"]:
                    raise ValueError("Invalid API response: missing choices")

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

                return {
                    "content": content,
                    "usage": usage,
                    "model": data.get("model", model),
                }
        except httpx.HTTPStatusError as e:
            last_error = e
            error_detail = f"HTTP {e.response.status_code}"
            try:
                error_body = e.response.json()
                error_detail += f": {error_body}"
            except Exception:
                error_detail += f": {e.response.text[:200]}"
            logger.warning(
                f"LLM API call failed (attempt {attempt + 1}/{retries}) at {api_url}: {error_detail}",
            )
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except (httpx.HTTPError, ValueError) as e:
            last_error = e
            logger.warning(
                f"LLM API call failed (attempt {attempt + 1}/{retries}) at {api_url}: {e}",
            )
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    error_msg = f"Failed to call LLM API at {api_url} after {retries} attempts"
    if last_error:
        error_msg += f": {last_error}"
    raise httpx.HTTPError(error_msg) from last_error


def call_llm_sync(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    retries: int = 3,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for call_llm.

    Args:
        model: Model name
        prompt: User prompt text
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        system_prompt: Optional system prompt
        retries: Number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        Dictionary with 'content' and 'usage'
    """
    return asyncio.run(
        call_llm(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            retries=retries,
            timeout=timeout,
        ),
    )

