import asyncio
import logging
import os
from urllib.parse import urlsplit
from typing import Any, Dict, List, Optional

import openai
from openai import AzureOpenAI  # <-- add this import

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(self, model_cfg: Optional[dict] = None):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        # Optional: pull api_version from config or env (fallback to a sane default)
        self.api_version = getattr(model_cfg, "api_version", None) or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2025-01-01-preview"
        )

        # Detect Azure vs OpenAI
        api_base_lower = (self.api_base or "").lower()
        self.is_azure = "openai.azure.com" in api_base_lower

        max_retries = self.retries if self.retries is not None else 0

        if self.is_azure:
            # Extract resource root (scheme + host) from any api_base (even if it includes /openai/deployments/…)
            parts = urlsplit(self.api_base)
            azure_endpoint = f"{parts.scheme}://{parts.netloc}"

            # Use the Azure client so it appends ?api-version=… correctly
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                timeout=self.timeout,
                max_retries=max_retries,
            )
        else:
            # Standard OpenAI client
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=max_retries,
            )

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Reasoning model prefixes (apply for both OpenAI and Azure)
        REASONING_PREFIXES = (
            "o1", "o1-", "o3", "o3-", "o4-",  # O-series
            "gpt-5", "gpt-5-",                # GPT-5 series
            "gpt-oss-120b", "gpt-oss-20b",
        )
        model_lower = str(self.model).lower()
        is_reasoning_model = model_lower.startswith(REASONING_PREFIXES)

        if is_reasoning_model:
            params = {
                "model": self.model,  # In Azure: this is the *deployment name*
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort

        # Seed (avoid on Google openai-compatible endpoint)
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None and self.api_base != "https://generativelanguage.googleapis.com/v1beta/openai/":
            params["seed"] = seed

        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(**params))
        logger = logging.getLogger(__name__)
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {resp.choices[0].message.content}")
        return resp.choices[0].message.content
