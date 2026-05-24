from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Protocol


LLMProvider = Literal["openai", "claude-agent"]


# ---------------------------------------------------------------------------
# LLM interaction logging (prompt + response)
# ---------------------------------------------------------------------------
_log_counter = 0


def _log_llm_interaction(*, provider: str, model_name: str, prompt: str, response: str, error: str | None = None) -> None:
    """Persist every prompt/response pair to disk for deterministic debugging."""
    global _log_counter
    _log_counter += 1

    # Allow caller to override via env; fall back to a project-local default.
    log_dir = Path(
        os.getenv("KSEARCH_LLM_LOG_DIR")
        or os.path.join(os.getcwd(), ".ksearch-output-mqa", "llm_logs")
    ).expanduser().resolve()

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return  # Silently skip if we can't create the directory.

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_model = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)[:48]
    filename = f"{ts}_{_log_counter:04d}_{provider}_{safe_model}.json"
    path = log_dir / filename

    payload = {
        "timestamp_utc": ts,
        "provider": provider,
        "model_name": model_name,
        "prompt": str(prompt or ""),
        "response": str(response or ""),
    }
    if error is not None:
        payload["error"] = error

    try:
        import json
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # Logging must never break the caller.


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
        try:
            if self.model_name.startswith("gpt-5") or self.model_name.startswith("o3"):
                response = self.client.responses.create(
                    model=self.model_name,
                    input=prompt,
                    reasoning={"effort": self.reasoning_effort},
                )
                result = str(getattr(response, "output_text", "") or "").strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                choice0 = response.choices[0] if getattr(response, "choices", None) else None
                message = getattr(choice0, "message", None)
                result = str(getattr(message, "content", "") or "").strip()
            _log_llm_interaction(provider="openai", model_name=self.model_name, prompt=prompt, response=result)
            return result
        except Exception as exc:
            _log_llm_interaction(
                provider="openai", model_name=self.model_name, prompt=prompt, response="", error=str(exc)
            )
            raise


def _default_claude_agent_max_turns() -> Optional[int]:
    """Default unlimited (None). Per SDK design `max_turns` counts conversation
    rounds, not chunks of long output, so single-completion use cases should
    not cap it. Override via env CLAUDE_AGENT_MAX_TURNS for diagnostics.
    """
    raw = os.getenv("CLAUDE_AGENT_MAX_TURNS")
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _default_claude_agent_thinking_enabled() -> bool:
    raw = os.getenv("CLAUDE_AGENT_THINKING", "").strip().lower()
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return True
    return False


def _default_claude_agent_timeout_seconds() -> float:
    for name in ("KSEARCH_LLM_TIMEOUT_SECONDS", "CLAUDE_AGENT_TIMEOUT_SECONDS"):
        raw = os.getenv(name)
        if raw is None or not str(raw).strip():
            continue
        try:
            value = float(str(raw).strip())
        except ValueError:
            continue
        if value > 0:
            return value

    raw_ms = os.getenv("API_TIMEOUT_MS")
    if raw_ms is not None and str(raw_ms).strip():
        try:
            value_ms = float(str(raw_ms).strip())
        except ValueError:
            value_ms = 0.0
        if value_ms > 0:
            return value_ms / 1000.0

    return 600.0


@dataclass
class ClaudeAgentLLMClient:
    model_name: str
    max_turns: Optional[int] = field(default_factory=_default_claude_agent_max_turns)
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=lambda: ["Bash"])
    thinking_enabled: bool = field(default_factory=_default_claude_agent_thinking_enabled)
    timeout_seconds: float = field(default_factory=_default_claude_agent_timeout_seconds)

    def generate(self, prompt: str) -> str:
        try:
            import claude_agent_sdk  # type: ignore
        except ImportError as exc:
            _log_llm_interaction(
                provider="claude-agent", model_name=self.model_name, prompt=prompt, response="", error=str(exc)
            )
            raise RuntimeError(
                "Claude Agent SDK provider requires the 'claude-agent-sdk' package. "
                "Install it with: pip install claude-agent-sdk"
            ) from exc

        async def _run_query() -> str:
            options_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "allowed_tools": list(self.allowed_tools),
                "disallowed_tools": list(self.disallowed_tools),
                "permission_mode": "bypassPermissions",
            }
            if self.max_turns is not None:
                options_kwargs["max_turns"] = self.max_turns
            if not self.thinking_enabled:
                options_kwargs["thinking"] = {"type": "disabled"}
            options = claude_agent_sdk.ClaudeAgentOptions(**options_kwargs)
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

        try:
            result = self._run_async(_run_query)
            _log_llm_interaction(provider="claude-agent", model_name=self.model_name, prompt=prompt, response=result)
            return result
        except Exception as exc:
            _log_llm_interaction(
                provider="claude-agent", model_name=self.model_name, prompt=prompt, response="", error=str(exc)
            )
            raise

    def _run_async(self, coro_factory: Any) -> str:
        timeout = float(self.timeout_seconds or 0)
        started = time.monotonic()

        async def _timed_run() -> str:
            if timeout <= 0:
                return await coro_factory()
            return await asyncio.wait_for(coro_factory(), timeout=timeout)

        def _timeout_error(exc: BaseException) -> TimeoutError:
            return TimeoutError(
                f"Claude Agent SDK provider timed out after {timeout:g}s. "
                "Set KSEARCH_LLM_TIMEOUT_SECONDS or API_TIMEOUT_MS to adjust this limit."
            )

        def _looks_like_timeout_cancel(exc: BaseException) -> bool:
            if timeout <= 0:
                return False
            elapsed = time.monotonic() - started
            text = str(exc)
            return elapsed >= (timeout * 0.9) and "exit code 143" in text

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(_timed_run())
            except TimeoutError as exc:
                raise _timeout_error(exc) from exc
            except RuntimeError as exc:
                if _looks_like_timeout_cancel(exc):
                    raise _timeout_error(exc) from exc
                raise

        def _runner() -> str:
            return asyncio.run(_timed_run())

        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                return executor.submit(_runner).result()
            except TimeoutError as exc:
                raise _timeout_error(exc) from exc
            except RuntimeError as exc:
                if _looks_like_timeout_cancel(exc):
                    raise _timeout_error(exc) from exc
                raise

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
