"""Shared constants and small helpers for the output contract.

Kept in one place so the CLI, rules engine, LLM layer, and evaluation
script all agree on the exact allowed values from problem_statement.md.
"""
from __future__ import annotations

ACTIONS = ("notify", "digest", "mute")

MESSAGE_TYPES = (
    "personal",
    "urgent",
    "event",
    "payment",
    "business_update",
    "promotion",
    "greeting",
    "forward",
    "spam",
    "scam",
    "unknown",
)

OUTPUT_COLUMNS = (
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
)

INPUT_COLUMNS = (
    "message_id",
    "user_id",
    "conversation_type",
    "group_id",
    "business_id",
    "sender_user_id",
    "created_at",
    "message_text",
    "media_type",
    "media_id",
    "forwarded_count",
)


def clamp_confidence(value: float) -> float:
    """Clamp to [0, 1] and round to 2 decimals for a clean, calibrated-looking output."""
    value = max(0.0, min(1.0, float(value)))
    return round(value, 2)


def validate_action(action: str) -> str:
    if action not in ACTIONS:
        raise ValueError(f"Invalid action {action!r}; must be one of {ACTIONS}")
    return action


def validate_message_type(message_type: str) -> str:
    if message_type not in MESSAGE_TYPES:
        raise ValueError(f"Invalid message_type {message_type!r}; must be one of {MESSAGE_TYPES}")
    return message_type


def validate_output_csv(output_path: str, expected_message_ids) -> None:
    """Final pre-submission validation of output.csv against the challenge's
    checklist: exact column order, exactly one row per input message_id, no
    extra ids, allowed action/message_type values, numeric confidence in
    [0, 1], non-empty reasons, and a well-formed evidence field. Raises
    ValueError with a clear message on the first violation found.
    """
    import csv

    expected = set(expected_message_ids)
    with open(output_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != list(OUTPUT_COLUMNS):
            raise ValueError(f"output.csv header {header!r} does not match required column order {list(OUTPUT_COLUMNS)!r}")
        seen = []
        for row in reader:
            if len(row) != len(OUTPUT_COLUMNS):
                raise ValueError(f"malformed row (wrong column count): {row!r}")
            rec = dict(zip(OUTPUT_COLUMNS, row))
            seen.append(rec["message_id"])
            validate_action(rec["action"])
            validate_message_type(rec["message_type"])
            try:
                conf = float(rec["confidence"])
            except ValueError as exc:
                raise ValueError(f"non-numeric confidence for {rec['message_id']}: {rec['confidence']!r}") from exc
            if not (0.0 <= conf <= 1.0):
                raise ValueError(f"confidence out of [0,1] for {rec['message_id']}: {conf}")
            if not rec["reason"].strip():
                raise ValueError(f"empty reason for {rec['message_id']}")
            if "\n" in rec["reason"] or "\r" in rec["reason"]:
                raise ValueError(f"reason contains a line break for {rec['message_id']}")
            ev = rec["evidence_message_ids"].strip()
            if ev != "none" and not ev:
                raise ValueError(f"empty (not 'none') evidence_message_ids for {rec['message_id']}")

    seen_set = set(seen)
    if len(seen) != len(seen_set):
        dupes = sorted({m for m in seen if seen.count(m) > 1})
        raise ValueError(f"duplicate message_id rows in output.csv: {dupes}")
    missing = expected - seen_set
    if missing:
        raise ValueError(f"output.csv is missing {len(missing)} message_id(s) from the input, e.g. {sorted(missing)[:5]}")
    extra = seen_set - expected
    if extra:
        raise ValueError(f"output.csv has {len(extra)} extra message_id(s) not present in the input, e.g. {sorted(extra)[:5]}")
