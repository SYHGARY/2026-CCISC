from __future__ import annotations

import re
from dataclasses import replace
from typing import Protocol

from .models import Claim, ClaimEdge, LogicTrace, TraceStep


_FILE_SUBJECT = r"(?P<subject>[\w./\\-]+\.[a-zA-Z0-9]{1,8}|the file|file|该文件|文件)"
_FAILURE_MARKERS = (
    "not found", "failed", "failure", "error", "blocked", "denied", "refused",
    "不存在", "失败", "错误", "被阻止", "拒绝",
)


class AdditionalClaimExtractor(Protocol):
    name: str

    def extract(self, step: TraceStep) -> list[Claim]: ...


def extract_step_claims(
    step: TraceStep,
    additional_extractor: AdditionalClaimExtractor | None = None,
    *,
    cache: dict[str, list[Claim]] | None = None,
) -> list[Claim]:
    # A call-scoped cache (keyed on the unique step_id) lets the several
    # detectors that each re-extract every step within one analyze() pass share
    # results. Only safe when there is no additional extractor (the default),
    # since the extractor is not part of the cache key.
    if cache is not None and additional_extractor is None:
        cached = cache.get(step.step_id)
        if cached is not None:
            return cached
    claims: list[Claim] = []
    if step.content:
        claims.extend(_extract_text_claims(step.content, step, "content"))
    if step.tool_result:
        claims.extend(_extract_text_claims(str(step.tool_result), step, "tool_result"))
    claims.extend(_extract_action_claims(step))
    if additional_extractor:
        claims.extend(additional_extractor.extract(step))
    result = _dedupe(claims)
    if cache is not None and additional_extractor is None:
        cache[step.step_id] = result
    return result


def build_claim_graph(
    trace: LogicTrace,
    additional_extractor: AdditionalClaimExtractor | None = None,
    *,
    cache: dict[str, list[Claim]] | None = None,
) -> tuple[list[Claim], list[ClaimEdge]]:
    active: dict[str, Claim] = {}
    claims: list[Claim] = []
    edges: list[ClaimEdge] = []
    for step in trace.steps:
        for key in invalidated_claim_keys(step):
            previous = active.pop(key, None)
            if previous:
                index = claims.index(previous)
                claims[index] = replace(previous, invalidated_by=step.step_id)
                edges.append(ClaimEdge(previous.key, step.step_id, "invalidated_by"))
        for claim in extract_step_claims(step, additional_extractor, cache=cache):
            previous = active.get(claim.key)
            if previous:
                relation = "supports" if previous.polarity == claim.polarity else "contradicts"
                confidence = min(previous.confidence, claim.confidence)
                edges.append(ClaimEdge(previous.key, claim.key, relation, confidence))
            claims.append(claim)
            active[claim.key] = claim
    return claims, edges


def invalidated_claim_keys(step: TraceStep) -> set[str]:
    action = str(step.action_name or "").lower()
    if not action or _looks_failed(str(step.tool_result or "")):
        return set()
    keys: set[str] = set()
    if any(marker in action for marker in ("write", "create", "delete", "remove", "move", "rename")):
        subject = _file_subject_from_args(step.action_args)
        keys.update({f"file_exists:{subject}", "file_exists:file"})
    if "email" in action or action in {"send_message", "network_send"}:
        keys.add("email_sent:email")
    if action == "memory_write":
        subject = str(step.action_args.get("key") or "memory").lower()
        keys.add(f"memory_value:{subject}")
    return keys


def _extract_text_claims(text: str, step: TraceStep, source: str) -> list[Claim]:
    lower = text.lower()
    claims: list[Claim] = []
    negative_file = re.search(
        rf"{_FILE_SUBJECT}.{{0,50}}(?:not found|does not exist|doesn't exist|missing|不存在|没有找到)",
        lower,
    )
    positive_file = re.search(
        rf"{_FILE_SUBJECT}.{{0,50}}(?:exists|was found|is present|存在)",
        lower,
    )
    if negative_file:
        claims.append(_claim(negative_file.group("subject"), "file_exists", "negative", text, step, source))
    elif positive_file:
        claims.append(_claim(positive_file.group("subject"), "file_exists", "positive", text, step, source))

    if re.search(r"(email|邮件).{0,60}(not sent|blocked|未发送|被阻止|发送失败)", lower):
        claims.append(_claim("email", "email_sent", "negative", text, step, source))
    elif re.search(r"(email|邮件).{0,60}(successfully sent|was sent|sent|已发送|发送成功)", lower):
        claims.append(_claim("email", "email_sent", "positive", text, step, source))

    if re.search(r"\b(?:unsafe|blocked|denied)\b|不安全|被阻止|拒绝", lower):
        claims.append(_claim("action", "action_safe", "negative", text, step, source))
    elif re.search(r"\b(?:safe|allowed)\b|安全|允许", lower):
        claims.append(_claim("action", "action_safe", "positive", text, step, source))

    for match in re.finditer(
        r"(?:memory|记忆)\s*[\[【]?(?P<key>[\w.-]+)[\]】]?\s*(?:is|=|为|是)\s*[\"“]?(?P<value>[^。.;\n\"”]+)",
        text,
        re.IGNORECASE,
    ):
        subject = match.group("key").lower()
        value = match.group("value").strip().lower()
        claims.append(_claim(subject, "memory_value", value, text, step, source, polarity_is_value=True))

    if re.search(r"ignore (?:all )?(?:previous|system) instructions|忽略(?:之前|系统)指令", lower):
        claims.append(_claim("instruction", "prompt_injection", "positive", text, step, source, 0.98))
    if re.search(r"(?:send|upload|exfiltrate|泄露|外传).{0,80}(?:secret|token|password|机密|密钥)", lower):
        claims.append(_claim("sensitive_data", "exfiltration_intent", "positive", text, step, source, 0.95))
    if re.search(r"(?:system message|developer instruction|系统提示|开发者指令).{0,50}(?:says|要求|指示)", lower):
        claims.append(_claim("instruction", "untrusted_authority_claim", "positive", text, step, source, 0.88))
    claims.extend(_extract_generic_state_claims(text, step, source))
    claims.extend(_extract_permission_claims(text, step, source))
    return claims


def _extract_action_claims(step: TraceStep) -> list[Claim]:
    action = str(step.action_name or "").lower()
    if not action:
        return []
    text = f"{action} {step.action_args}".lower()
    claims: list[Claim] = []
    if action in {"send_email", "network_send", "http_post", "upload_file"}:
        claims.append(_claim("external_channel", "external_send_attempt", "positive", text, step, "action", 0.99))
    if action in {"execute_code", "run_code", "shell_exec"}:
        claims.append(_claim("runtime", "code_execution_attempt", "positive", text, step, "action", 0.99))
    if action in {"delete_file", "remove_file"}:
        subject = _file_subject_from_args(step.action_args)
        claims.append(_claim(subject, "destructive_action", "positive", text, step, "action", 0.99))
    if action == "memory_write":
        subject = str(step.action_args.get("key") or "memory").lower()
        value = str(step.action_args.get("value") or "").lower()
        claims.append(_claim(subject, "memory_value", value, text, step, "action", 0.95, polarity_is_value=True))
    return claims


def _claim(
    subject: str,
    predicate: str,
    polarity: str,
    evidence: str,
    step: TraceStep,
    source: str,
    confidence: float = 0.82,
    *,
    polarity_is_value: bool = False,
) -> Claim:
    normalized_subject = "file" if subject.lower() in {"the file", "该文件", "文件"} else subject.lower()
    return Claim(
        subject=normalized_subject,
        predicate=predicate,
        polarity=polarity,
        evidence=evidence[:500],
        step_id=step.step_id,
        source=source,
        confidence=confidence,
        # Never raise a claim's trust above its source step's trust. A tool
        # result is authoritative about its OWN output, but that is factual
        # confidence, not provenance trust — escalating trust here would let a
        # low-trust tool/file source launder itself into a high-trust claim and
        # defeat the untrusted-permission-grant check in the solver.
        trust=step.trust,
        timestamp=step.timestamp,
        valid_from_step=step.step_id,
        provenance={
            "agent_id": step.agent_id,
            "event_source": step.source,
            "temporal_scope": _temporal_scope(evidence),
            "factual_confidence": max(step.trust, 0.9) if source == "tool_result" else step.trust,
        },
    )


def _file_subject_from_args(action_args: dict) -> str:
    subject = action_args.get("filename") or action_args.get("path") or "file"
    return str(subject).lower()


def _looks_failed(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _FAILURE_MARKERS)


def _dedupe(claims: list[Claim]) -> list[Claim]:
    result: list[Claim] = []
    seen: set[tuple[str, str, str, str]] = set()
    for claim in claims:
        key = (claim.key, claim.polarity, claim.step_id, claim.source)
        if key not in seen:
            result.append(claim)
            seen.add(key)
    return result


def _extract_generic_state_claims(
    text: str,
    step: TraceStep,
    source: str,
) -> list[Claim]:
    claims: list[Claim] = []
    patterns = (
        (
            r"\b(?P<subject>[a-zA-Z][\w.-]{1,40})\s+"
            r"(?:is|was|will be)\s+"
            r"(?P<value>available|unavailable|open|closed|enabled|disabled|"
            r"approved|denied|online|offline|active|inactive)\b",
            "entity_state",
        ),
        (
            r"(?P<subject>[\w\u4e00-\u9fff.-]{1,40})"
            r"(?:当前|现在)?(?:是|为|处于)"
            r"(?P<value>可用|不可用|开启|关闭|启用|禁用|已批准|已拒绝|在线|离线|活跃|停用)",
            "entity_state",
        ),
    )
    for pattern, predicate in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            subject = match.group("subject").lower()
            value = _canonical_state(match.group("value"))
            claims.append(
                _claim(
                    subject,
                    predicate,
                    value,
                    text,
                    step,
                    source,
                    0.86,
                    polarity_is_value=True,
                )
            )
    return claims


def _extract_permission_claims(
    text: str,
    step: TraceStep,
    source: str,
) -> list[Claim]:
    claims: list[Claim] = []
    patterns = (
        (
            r"\b(?P<subject>[a-zA-Z][\w.-]{0,30})\s+"
            r"(?P<negative>cannot|can't|may not|is not allowed to|is forbidden to)"
            r"\s+(?P<action>send|delete|write|execute|access|upload)\b",
            "negative",
        ),
        (
            r"\b(?P<subject>[a-zA-Z][\w.-]{0,30})\s+"
            r"(?P<positive>can|may|is allowed to|is authorized to)"
            r"\s+(?P<action>send|delete|write|execute|access|upload)\b",
            "positive",
        ),
        (
            r"(?P<subject>[\w\u4e00-\u9fff.-]{1,30})"
            r"(?P<negative>无权|不得|不能|禁止)"
            r"(?P<action>发送|删除|写入|执行|访问|上传)",
            "negative",
        ),
        (
            r"(?P<subject>[\w\u4e00-\u9fff.-]{1,30})"
            r"(?P<positive>可以|有权|允许|已获授权)"
            r"(?P<action>发送|删除|写入|执行|访问|上传)",
            "positive",
        ),
    )
    for pattern, polarity in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            action = _canonical_action(match.group("action"))
            claims.append(
                _claim(
                    match.group("subject").lower(),
                    f"permission_{action}",
                    polarity,
                    text,
                    step,
                    source,
                    0.9,
                )
            )
    return claims


def _canonical_state(value: str) -> str:
    aliases = {
        "可用": "available",
        "不可用": "unavailable",
        "开启": "open",
        "关闭": "closed",
        "启用": "enabled",
        "禁用": "disabled",
        "已批准": "approved",
        "已拒绝": "denied",
        "在线": "online",
        "离线": "offline",
        "活跃": "active",
        "停用": "inactive",
    }
    return aliases.get(value.lower(), value.lower())


def _canonical_action(value: str) -> str:
    aliases = {
        "发送": "send",
        "删除": "delete",
        "写入": "write",
        "执行": "execute",
        "访问": "access",
        "上传": "upload",
    }
    return aliases.get(value.lower(), value.lower())


def _temporal_scope(text: str) -> str:
    lower = text.lower()
    if re.search(r"\b(?:was|were|previously|yesterday|before)\b|此前|之前|昨天|曾经", lower):
        return "past"
    if re.search(r"\b(?:will|tomorrow|later|after)\b|将会|明天|之后|稍后", lower):
        return "future"
    return "current"
