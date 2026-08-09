"""Retrieve historical message_ids (from dataset/message_history.csv) that
justify a routing decision, joined with dataset/message_events.csv to see how
the user actually reacted last time (opened/replied/dismissed/muted/reported).

Matching strategy (simple, explainable, no embeddings required):
  1. Restrict to history rows belonging to the same receiving user.
  2. Prefer rows from the exact same sender_user_id / business_id / group_id.
  3. Rank by a mix of (a) token-overlap text similarity with the incoming
     message, (b) recency, and (c) whether the row has a recorded reaction
     (rows with an event are more useful evidence than ones without).
  4. Return the top N message_ids, semicolon-separated, or "none".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .data_loader import Dataset

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _parse_dt(value: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except Exception:
            continue
    return None


@dataclass
class EvidenceItem:
    message_id: str
    score: float
    row: Dict[str, str]


def find_evidence(ds: Dataset, message: Dict[str, str], max_items: int = 2) -> List[EvidenceItem]:
    user_id = message.get("user_id")
    sender_user_id = message.get("sender_user_id") or None
    business_id = message.get("business_id") or None
    group_id = message.get("group_id") or None
    text = message.get("message_text") or ""
    incoming_tokens = _tokens(text)
    incoming_dt = _parse_dt(message.get("created_at", ""))

    candidates: Dict[str, Dict[str, str]] = {}
    for pool, relation_weight in (
        (ds.history_by_sender.get(sender_user_id, []) if sender_user_id else [], 1.0),
        (ds.history_by_business.get(business_id, []) if business_id else [], 1.0),
        (ds.history_by_group.get(group_id, []) if group_id else [], 0.6),
        (ds.history_by_user.get(user_id, []) if user_id else [], 0.3),
    ):
        for row in pool:
            if row.get("user_id") != user_id:
                continue
            mid = row.get("message_id")
            if not mid or mid in candidates:
                continue
            candidates[mid] = row

    scored: List[EvidenceItem] = []
    for mid, row in candidates.items():
        same_sender = sender_user_id and row.get("sender_user_id") == sender_user_id
        same_business = business_id and row.get("business_id") == business_id
        same_group = group_id and row.get("group_id") == group_id

        relation_score = 0.0
        if same_sender or same_business:
            relation_score = 1.0
        elif same_group:
            relation_score = 0.5
        else:
            relation_score = 0.2

        overlap = 0.0
        row_tokens = _tokens(row.get("message_text", ""))
        if incoming_tokens and row_tokens:
            inter = incoming_tokens & row_tokens
            union = incoming_tokens | row_tokens
            overlap = len(inter) / max(1, len(union))

        recency = 0.0
        row_dt = _parse_dt(row.get("created_at", ""))
        if incoming_dt and row_dt:
            days = max(0.0, (incoming_dt - row_dt).total_seconds() / 86400.0)
            recency = 1.0 / (1.0 + days / 14.0)

        has_reaction = 1.0 if ds.get_event(mid) else 0.0

        score = 0.45 * relation_score + 0.3 * overlap + 0.15 * recency + 0.1 * has_reaction
        scored.append(EvidenceItem(message_id=mid, score=score, row=row))

    scored.sort(key=lambda item: item.score, reverse=True)
    # keep only genuinely relevant evidence - avoid padding with weak matches
    relevant = [item for item in scored if item.score >= 0.35]
    return relevant[:max_items]


def format_evidence_ids(items: List[EvidenceItem]) -> str:
    if not items:
        return "none"
    return ";".join(item.message_id for item in items)
