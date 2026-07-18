from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_deepseek_batch_experiment import load_cases, needs_sandbox_fixture

DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "deepseek_official_benchmark_560.json"
DEFAULT_STAGE20_RUN = PROJECT_ROOT / "outputs" / "deepseek_official560_dry_run.json"
DEFAULT_ROOT_JSON = PROJECT_ROOT / "outputs" / "stage21_case_level_root_cause.json"
DEFAULT_ROOT_MD = PROJECT_ROOT / "outputs" / "stage21_case_level_root_cause.md"
DEFAULT_SEED_JSON = PROJECT_ROOT / "outputs" / "stage21_seed_files_fidelity_audit.json"
DEFAULT_SEED_MD = PROJECT_ROOT / "outputs" / "stage21_seed_files_fidelity_audit.md"
FALSE_POSITIVE_IDS = ("norm_store_language", "hard_store_non_secret_preference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 21 case-level analysis reports.")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--stage20-run", default=str(DEFAULT_STAGE20_RUN))
    parser.add_argument("--root-json-output", default=str(DEFAULT_ROOT_JSON))
    parser.add_argument("--root-md-output", default=str(DEFAULT_ROOT_MD))
    parser.add_argument("--seed-json-output", default=str(DEFAULT_SEED_JSON))
    parser.add_argument("--seed-md-output", default=str(DEFAULT_SEED_MD))
    return parser.parse_args()


def build_reports(benchmark_path: Path, stage20_run_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    stage20 = json.loads(stage20_run_path.read_text(encoding="utf-8"))
    raw_cases = list(benchmark.get("cases", []))
    normalized_cases = [
        item
        for rows in load_cases(benchmark_path).values()
        for item in rows
    ]
    root = build_root_cause(raw_cases, stage20)
    seed = build_seed_audit(raw_cases, normalized_cases)
    return root, seed


def build_root_cause(raw_cases: list[dict[str, Any]], stage20: dict[str, Any]) -> dict[str, Any]:
    raw_by_id = {str(item.get("case_id")): item for item in raw_cases}
    records_by_id = {str(item.get("id")): item for item in stage20.get("records", [])}
    findings = []
    for case_id in FALSE_POSITIVE_IDS:
        record = records_by_id[case_id]
        guarded = record["guarded"]
        action_result_decision = first_action_result_decision(guarded)
        candidate = (guarded.get("candidate_actions") or [{}])[-1]
        risk = extract_risk_summary(action_result_decision)
        findings.append({
            "case_id": case_id,
            "sample_type": record.get("dataset"),
            "category": record.get("category"),
            "attack_surface": record.get("attack_surface", []),
            "user_prompt": raw_by_id.get(case_id, {}).get("user_prompt", record.get("goal", "")),
            "candidate_action": candidate.get("action"),
            "candidate_args": candidate.get("args", {}),
            "memory_key": (candidate.get("args") or {}).get("key"),
            "memory_value": (candidate.get("args") or {}).get("value"),
            "expected_guard_behavior": raw_by_id.get(case_id, {}).get("expected_guard_behavior", []),
            "first_blocking_phase": "action-result",
            "first_blocking_rule": "probabilistic_sparse_state_fallback",
            "decision": action_result_decision.get("action"),
            "enforcement": action_result_decision.get("alternative_action"),
            "risk": risk,
            "root_cause": (
                "The before-action memory_write was user-authorized, but the after-action event "
                "was evaluated as source=tool. The sparse-state fallback treated the memory write "
                "as untrusted even though the key/value are a benign language preference."
            ),
            "classification": "guard_strategy_false_positive",
            "should_quarantine": False,
        })

    baseline_success_ids = [
        str(item.get("id"))
        for item in stage20.get("records", [])
        if item.get("dataset") == "attack" and item.get("baseline", {}).get("attack_success")
    ]
    attack_records = [item for item in stage20.get("records", []) if item.get("dataset") == "attack"]
    baseline_asr = (
        round(len(baseline_success_ids) / len(attack_records), 6)
        if attack_records
        else 0.0
    )
    return {
        "stage": "Stage 21",
        "source_run": str(DEFAULT_STAGE20_RUN),
        "false_positive_case_ids": list(FALSE_POSITIVE_IDS),
        "findings": findings,
        "baseline_asr_low_root_cause": {
            "stage20_baseline_asr": baseline_asr,
            "baseline_success_case_ids": baseline_success_ids,
            "judgment": "fixture_issue",
            "explanation": (
                "The low deterministic baseline ASR is mainly a dry-run fidelity issue: many generated "
                "official cases had labels and prompts but no sandbox seed_files, while the deterministic "
                "provider exposes attacks most clearly through low-trust file or tool observations. "
                "Stage 21 fixture loading improves that observation fidelity, but generated output-only "
                "or jailbreak cases can still have attack_success_tools=[]; those cases need a later "
                "metric-design pass before baseline ASR can be interpreted as attack strength."
            ),
        },
        "issue_classification": {
            "guard_strategy_issue": list(FALSE_POSITIVE_IDS),
            "fixture_issue": "generated official cases missing sandbox seed_files",
            "statistics_issue": "some generated output-only/jailbreak cases do not have tool-level attack_success criteria",
            "data_quality_issue": [],
        },
    }


def first_action_result_decision(guarded: dict[str, Any]) -> dict[str, Any]:
    for decision in guarded.get("guard_decisions", []):
        if "action-result" in str(decision.get("event_id", "")):
            return decision
    return {}


def extract_risk_summary(decision: dict[str, Any]) -> dict[str, Any]:
    reason = " ".join(str(item) for item in decision.get("reasons", []))
    return {
        "reason": reason,
        "fallback": "稀疏状态回退" in reason or "sparse" in reason.lower(),
        "features_observed_in_stage20": {
            "memory_write": True,
            "permission_granted": True,
            "memory_trusted": False,
        },
    }


def build_seed_audit(
    raw_cases: list[dict[str, Any]],
    normalized_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_by_id = {str(item.get("case_id")): item for item in raw_cases}
    raw_needing = [item for item in raw_cases if needs_sandbox_fixture(to_runner_case(item))]
    raw_missing = [item for item in raw_needing if not item.get("seed_files")]
    normalized_needing = [item for item in normalized_cases if needs_sandbox_fixture(item)]
    normalized_missing = [item for item in normalized_needing if not item.get("seed_files")]
    generated = [
        item
        for item in normalized_needing
        if (
            not item.get("is_core_case")
            and not raw_by_id.get(str(item.get("id")), {}).get("seed_files")
            and item.get("seed_files")
        )
    ]
    core_preserved = [
        item
        for item in normalized_needing
        if (
            item.get("is_core_case")
            and not raw_by_id.get(str(item.get("id")), {}).get("seed_files")
            and item.get("seed_files")
        )
    ]
    unsafe_hits = [
        item.get("id")
        for item in generated
        if contains_forbidden_real_secret(item.get("seed_files") or {})
    ]
    return {
        "stage": "Stage 21",
        "total_cases": len(raw_cases),
        "raw_cases_requiring_fixture": len(raw_needing),
        "raw_missing_seed_files": len(raw_missing),
        "normalized_cases_requiring_fixture": len(normalized_needing),
        "normalized_missing_seed_files": len(normalized_missing),
        "core_fixture_preserved_from_core_dataset": len(core_preserved),
        "generated_fixture_case_count": len(generated),
        "generated_fixture_by_category": dict(Counter(str(item.get("category")) for item in generated)),
        "core_fixture_by_category": dict(Counter(str(item.get("category")) for item in core_preserved)),
        "normalized_missing_by_category": dict(Counter(str(item.get("category")) for item in normalized_missing)),
        "forbidden_real_secret_pattern_hits": unsafe_hits,
        "judgment": "passed" if not normalized_missing and not unsafe_hits else "needs_review",
        "notes": [
            "Fixtures are loaded at deterministic-run time; the official benchmark JSON is not rewritten.",
            "Core fixtures are preserved from data/deepseek_batch_cases.json during loader normalization.",
            "Stage 21 generated fixture contents use sandbox DEMO values and example.com addresses only.",
        ],
    }


def to_runner_case(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **item,
        "id": item.get("case_id"),
        "dataset": item.get("sample_type"),
        "goal": item.get("user_prompt", ""),
    }


def contains_forbidden_real_secret(seed_files: dict[str, str]) -> bool:
    text = json.dumps(seed_files, ensure_ascii=False).lower()
    forbidden = ("sk-", "ghp_", "xoxb-", "-----begin private key-----")
    return any(marker in text for marker in forbidden)


def render_root_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 21 Case-Level Root Cause",
        "",
        f"- false_positive_case_ids: `{report['false_positive_case_ids']}`",
        f"- baseline_asr_low_judgment: `{report['baseline_asr_low_root_cause']['judgment']}`",
        "",
        "## False Positive Cases",
        "",
    ]
    for finding in report["findings"]:
        lines.extend([
            f"### {finding['case_id']}",
            "",
            f"- sample_type: `{finding['sample_type']}`",
            f"- candidate_action: `{finding['candidate_action']}`",
            f"- memory_key: `{finding['memory_key']}`",
            f"- memory_value: `{finding['memory_value']}`",
            f"- first_blocking_phase: `{finding['first_blocking_phase']}`",
            f"- rule: `{finding['first_blocking_rule']}`",
            f"- decision: `{finding['decision']}`",
            f"- enforcement: `{finding['enforcement']}`",
            f"- benign_preference: `{not finding['should_quarantine']}`",
            f"- root_cause: {finding['root_cause']}",
            "",
        ])
    lines.extend([
        "## Baseline ASR Fidelity",
        "",
        f"- stage20_baseline_asr: `{report['baseline_asr_low_root_cause']['stage20_baseline_asr']}`",
        f"- baseline_success_case_ids: `{report['baseline_asr_low_root_cause']['baseline_success_case_ids']}`",
        f"- explanation: {report['baseline_asr_low_root_cause']['explanation']}",
        "",
    ])
    return "\n".join(lines)


def render_seed_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 21 Seed Files Fidelity Audit",
        "",
        f"- total_cases: `{report['total_cases']}`",
        f"- raw_cases_requiring_fixture: `{report['raw_cases_requiring_fixture']}`",
        f"- raw_missing_seed_files: `{report['raw_missing_seed_files']}`",
        f"- normalized_cases_requiring_fixture: `{report['normalized_cases_requiring_fixture']}`",
        f"- normalized_missing_seed_files: `{report['normalized_missing_seed_files']}`",
        f"- core_fixture_preserved_from_core_dataset: `{report['core_fixture_preserved_from_core_dataset']}`",
        f"- generated_fixture_case_count: `{report['generated_fixture_case_count']}`",
        f"- forbidden_real_secret_pattern_hits: `{report['forbidden_real_secret_pattern_hits']}`",
        f"- judgment: `{report['judgment']}`",
        "",
        "## Generated Fixtures By Category",
        "",
    ]
    for category, count in sorted(report["generated_fixture_by_category"].items()):
        lines.append(f"- {category}: `{count}`")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root, seed = build_reports(Path(args.benchmark), Path(args.stage20_run))
    write_json(Path(args.root_json_output), root)
    Path(args.root_md_output).write_text(render_root_markdown(root), encoding="utf-8")
    write_json(Path(args.seed_json_output), seed)
    Path(args.seed_md_output).write_text(render_seed_markdown(seed), encoding="utf-8")
    print(json.dumps({
        "root_cause_output": args.root_json_output,
        "seed_audit_output": args.seed_json_output,
        "seed_audit_judgment": seed["judgment"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
