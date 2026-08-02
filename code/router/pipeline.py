"""End-to-end orchestration: one incoming message -> one output.csv row.

Safety contract: the deterministic rules engine's hard safety calls
(prompt-injection / scam) are final. The optional LLM layer is only
consulted - and only allowed to change the verdict - when the rules engine
did NOT already hard-flag the message. This guarantees the system is safe
even if the LLM is unavailable, mis-configured, or itself manipulated by
adversarial message content.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .data_loader import Dataset, load_dataset
from .evidence import find_evidence, format_evidence_ids
from .features import build_features
from .llm import LLMReasoner
from .media import MediaAnalyzer, MediaCache
from .rules import Verdict, classify
from .schema import OUTPUT_COLUMNS, clamp_confidence, validate_action, validate_message_type, validate_output_csv

HARD_SAFETY_TAGS = {"prompt_injection_detected", "scam_keywords"}


@dataclass
class RouteResult:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str
    rules_verdict: Verdict
    final_verdict: Verdict
    used_llm: bool

    def to_row(self) -> Dict[str, str]:
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": f"{self.confidence:.2f}",
            "evidence_message_ids": self.evidence_message_ids,
        }


def _is_hard_safety(verdict: Verdict) -> bool:
    return verdict.action == "mute" and verdict.message_type == "scam" and bool(HARD_SAFETY_TAGS & set(verdict.tags))


def route_message(
    ds: Dataset,
    message: Dict[str, str],
    media_analyzer: MediaAnalyzer,
    llm_reasoner: Optional[LLMReasoner],
) -> RouteResult:
    media_type = (message.get("media_type") or "").strip()
    media_id = (message.get("media_id") or "").strip()
    media_result = None
    if media_type and media_id:
        file_path = ds.media_path(media_type, media_id)
        media_result = media_analyzer.analyze(media_type, media_id, file_path)

    features = build_features(ds, message, media_result)
    rules_verdict = classify(features)

    final_verdict = rules_verdict
    used_llm = False
    if llm_reasoner is not None and llm_reasoner.enabled and not _is_hard_safety(rules_verdict):
        llm_verdict = llm_reasoner.refine(features, rules_verdict)
        if llm_verdict is not None:
            final_verdict = llm_verdict
            used_llm = True

    evidence_items = find_evidence(ds, message)
    evidence_str = format_evidence_ids(evidence_items)

    action = validate_action(final_verdict.action)
    message_type = validate_message_type(final_verdict.message_type)
    confidence = clamp_confidence(final_verdict.confidence)

    return RouteResult(
        message_id=message["message_id"],
        action=action,
        message_type=message_type,
        reason=final_verdict.reason,
        confidence=confidence,
        evidence_message_ids=evidence_str,
        rules_verdict=rules_verdict,
        final_verdict=final_verdict,
        used_llm=used_llm,
    )


def run(
    dataset_dir: str,
    output_path: str,
    use_llm: bool = True,
    cache_path: Optional[str] = None,
    verbose: bool = False,
    limit: Optional[int] = None,
) -> List[RouteResult]:
    ds = load_dataset(dataset_dir)
    if not ds.messages:
        raise FileNotFoundError(
            f"No messages found under {dataset_dir}/messages.csv - check --dataset-dir points at the dataset folder."
        )

    cache_path = cache_path or os.path.join(dataset_dir, ".media_cache.json")
    cache = MediaCache(cache_path)
    media_analyzer = MediaAnalyzer(use_llm=use_llm, cache=cache, verbose=verbose)
    llm_reasoner = LLMReasoner(verbose=verbose) if use_llm else None

    results: List[RouteResult] = []
    rows = ds.messages if limit is None else ds.messages[:limit]
    for i, message in enumerate(rows):
        if verbose:
            print(f"[{i + 1}/{len(rows)}] routing {message.get('message_id')}...")
        results.append(route_message(ds, message, media_analyzer, llm_reasoner))

    cache.save()
    _write_output(output_path, results)
    if limit is None:
        validate_output_csv(output_path, (m["message_id"] for m in ds.messages))
        if verbose:
            print(f"[validate] output.csv passed schema/coverage validation ({len(results)} rows).")
    return results


def _write_output(output_path: str, results: List[RouteResult]) -> None:
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_row())
