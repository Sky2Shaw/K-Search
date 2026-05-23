from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol


LLMProvider = Literal["openai", "claude-agent"]


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Return model text for a single prompt."""


def normalize_llm_provider(provider: Optional[str]) -> LLMProvider:
    value = str(provider or "openai").strip().lower().replace("_", "-")
    if value in {"openai", "openai-compatible", "openai-compatible-api"}:
        return "openai"
    if value in {"claude", "claude-agent", "claude-agent-sdk"}:
        return "claude-agent"
    raise ValueError(
        f"Unsupported LLM provider {provider!r}; expected 'openai' or 'claude-agent'"
    )


@dataclass
class OpenAICompatibleLLMClient:
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    reasoning_effort: str = "medium"
    openai_module: Any = None
    client: Any = field(init=False)

    def __post_init__(self) -> None:
        key = self.api_key or os.getenv("LLM_API_KEY")
        if key is None or not str(key).strip():
            raise ValueError(
                "API key must be provided or set in LLM_API_KEY environment variable "
                "when llm_provider='openai'"
            )

        openai_mod = self.openai_module
        if openai_mod is None:
            try:
                import openai as openai_mod  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI-compatible provider requires the 'openai' Python package"
                ) from exc

        client_kwargs: dict[str, Any] = {"api_key": key}
        if self.base_url is not None:
            client_kwargs["base_url"] = self.base_url
        self.client = openai_mod.OpenAI(**client_kwargs)

    def generate(self, prompt: str) -> str:
        if self.model_name.startswith("gpt-5") or self.model_name.startswith("o3"):
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                reasoning={"effort": self.reasoning_effort},
            )
            return str(getattr(response, "output_text", "") or "").strip()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        choice0 = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice0, "message", None)
        return str(getattr(message, "content", "") or "").strip()


@dataclass
class ClaudeAgentLLMClient:
    model_name: str
    max_turns: int = 1
    allowed_tools: list[str] = field(default_factory=list)

    def generate(self, prompt: str) -> str:
        try:
            import claude_agent_sdk  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Claude Agent SDK provider requires the 'claude-agent-sdk' package. "
                "Install it with: pip install claude-agent-sdk"
            ) from exc

        async def _run_query() -> str:
            options = claude_agent_sdk.ClaudeAgentOptions(
                model=self.model_name,
                allowed_tools=list(self.allowed_tools),
                max_turns=self.max_turns,
            )
            assistant_chunks: list[str] = []
            final_text = ""
            try:
                async for message in claude_agent_sdk.query(prompt=prompt, options=options):
                    is_result_message = hasattr(message, "result")
                    if is_result_message:
                        self._ensure_successful_result_message(message)
                    text = self._extract_message_text(message)
                    if not text:
                        continue
                    if is_result_message:
                        final_text = text
                    else:
                        assistant_chunks.append(text)
            except Exception as exc:
                raise RuntimeError(f"Claude Agent SDK provider failed: {exc}") from exc

            result = (final_text or "\n".join(assistant_chunks)).strip()
            if not result:
                raise RuntimeError("Claude Agent SDK returned empty text")
            return result

        return self._run_async(_run_query)

    @staticmethod
    def _run_async(coro_factory: Any) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro_factory())

        def _runner() -> str:
            return asyncio.run(coro_factory())

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_runner).result()

    @staticmethod
    def _ensure_successful_result_message(message: Any) -> None:
        is_error = bool(getattr(message, "is_error", False))
        subtype = getattr(message, "subtype", None)
        subtype_text = str(subtype) if subtype is not None else None
        if not is_error and (subtype_text is None or subtype_text == "success"):
            return

        result = getattr(message, "result", None)
        result_text = str(result).strip() if result is not None else ""
        details = []
        if subtype_text is not None:
            details.append(f"subtype={subtype_text}")
        if is_error:
            details.append("is_error=True")
        context = f" ({', '.join(details)})" if details else ""
        suffix = f": {result_text}" if result_text else ""
        raise RuntimeError(f"Claude Agent SDK returned error result{context}{suffix}")

    @classmethod
    def _extract_message_text(cls, message: Any) -> str:
        for attr in ("result", "text"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                item_text = cls._extract_content_item_text(item)
                if item_text:
                    parts.append(item_text)
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _extract_content_item_text(item: Any) -> str:
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            value = item.get("text")
            return str(value).strip() if value is not None else ""
        value = getattr(item, "text", None)
        return str(value).strip() if value is not None else ""


def build_llm_client(
    *,
    llm_provider: Optional[str],
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    reasoning_effort: str = "medium",
) -> LLMClient:
    provider = normalize_llm_provider(llm_provider)
    if provider == "openai":
        return OpenAICompatibleLLMClient(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
        )
    return ClaudeAgentLLMClient(model_name=model_name)
