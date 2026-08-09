"""Deterministic, dependency-free scoring engine.

This is the safety net of the whole system: it must produce a reasonable,
personalized routing decision for every message using nothing but the
structured dataset - no network, no API key, no LLM. The optional LLM layer
(``llm.py``) may refine wording/nuance on top of this, but it can never
downgrade a hard safety call made here (see ``pipeline.py``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .features import Features

NEGATION_NEAR_SCAM_RE = re.compile(
    r"(never ask (you )?for|will never ask|does not ask|do not ask|"
    r"we (will )?never (call|ask|request)|beware of (scam|fraud)|"
    r"safety advisory|fraud (alert|warning)|protect yourself)",
    re.IGNORECASE,
)


@dataclass
class Verdict:
    action: str
    message_type: str
    confidence: float
    reason: str
    tags: List[str] = field(default_factory=list)


def _is_scam_advisory_not_scam(features: Features) -> bool:
    """A verified business warning users *about* scams should not itself be muted as one."""
    return bool(NEGATION_NEAR_SCAM_RE.search(features.combined_text)) and features.business_verified


def classify(features: Features) -> Verdict:
    f = features
    tags: List[str] = []

    # ---- 1. Hard safety overrides (cannot be relaxed by anything else) ----
    if f.prompt_injection:
        tags.append("prompt_injection_detected")
        return Verdict(
            action="mute",
            message_type="scam",
            confidence=0.9,
            reason=(
                "The message embeds instructions trying to manipulate the routing decision "
                "(e.g. asking to force notify/confidence); the router ignores those instructions "
                "and treats the underlying content as a scam attempt."
            ),
            tags=tags,
        )

    scam_advisory = _is_scam_advisory_not_scam(f)
    if f.scam_keywords and not scam_advisory:
        risk_context = []
        if not f.business_verified and f.business is not None:
            risk_context.append("unverified business account")
        if f.domain_mismatch:
            risk_context.append("sender domain does not match the official brand domain")
        if f.young_domain:
            risk_context.append("the sending domain/account is very new")
        if f.business_report_rate > 0.2:
            risk_context.append("many other users have recently reported this sender")
        if f.business is None and f.sender_familiarity < 0.2:
            risk_context.append("first/rare contact from this sender")
        if f.hard_scam_signal:
            detail_lead = "classic OTP/verification/prize-claim pressure language"
        else:
            detail_lead = "a link/QR call-to-action from an untrustworthy sender"
        detail = "; ".join(risk_context) if risk_context else detail_lead
        tags.append("scam_keywords")
        return Verdict(
            action="mute",
            message_type="scam",
            confidence=0.87 if f.hard_scam_signal else 0.75,
            reason=f"Message uses urgent verification/account-block/payment-pressure language ({detail}); routed as a likely scam regardless of the user's usual engagement.",
            tags=tags,
        )

    # ---- 1b. Secondary safety override: unverified sender with an
    # unusually high recent user-report rate. This does not require any
    # specific scam phrase to match (it also covers voice/image messages
    # with no usable transcript), so it is a distinct, slightly softer
    # override from the explicit scam-language case above.
    if (
        f.business is not None
        and not f.business_verified
        and f.business_report_rate > 0.2
        and not scam_advisory
    ):
        tags.append("high_risk_unverified_business")
        return Verdict(
            action="mute",
            message_type="spam",
            confidence=0.8,
            reason=(
                "Sender is an unverified business account with a much higher recent "
                "user-report rate than comparable senders; muted as likely spam "
                "regardless of the user's usual engagement, since the content itself "
                "can't be fully verified as safe."
            ),
            tags=tags,
        )

    # ---- 2. message_type classification -----------------------------------
    message_type = _classify_type(f, scam_advisory)

    # ---- 3. value / risk scoring -------------------------------------------
    value = 0.0
    risk = 0.0
    reasons: List[str] = []

    if f.direct_mention:
        value += 0.45
        reasons.append("directly @-mentions this user")
    if f.urgency_signal:
        value += 0.35
        reasons.append("contains urgent/time-sensitive language")
    if message_type in ("payment", "business_update", "event") and f.user_has_relevant_business_history:
        if f.actionable_signal:
            bonus = {"payment": 0.4, "business_update": 0.35, "event": 0.35}[message_type]
            reasons.append("matches a recent order/payment/booking relationship with this business and needs action soon")
        else:
            bonus = {"payment": 0.3, "business_update": 0.15, "event": 0.15}[message_type]
            reasons.append("matches the user's recent activity with this business")
        value += bonus
    elif message_type == "event":
        value += 0.15
        reasons.append("shares an operational/event update")
    if f.sender_familiarity > 0.5 and f.message.get("conversation_type") == "personal":
        value += 0.15
        reasons.append("from a contact the user usually engages with")
    if f.is_admin_sender and f.message.get("conversation_type") == "group":
        value += 0.1
        reasons.append("sent by a group admin")
        if message_type == "event" and f.admin_event_signal and (f.group or {}).get("group_type") in ("school_group", "society"):
            value += 0.2
            reasons.append("a same-context admin posted a formal school/society notice")

    if f.repetition_penalty > 0.3:
        risk += f.repetition_penalty
        reasons.append("similar past messages from this sender were dismissed/muted/reported")
    if f.group_muted and not f.direct_mention:
        risk += 0.45
        reasons.append("the user has muted this group and is not directly addressed")
    if message_type == "promotion" and f.user_opted_out_promo:
        risk += 0.5
        reasons.append("the user has opted out of promotions from this business")
    if message_type in ("greeting", "forward") and f.forwarded_count >= 3:
        risk += 0.25
        reasons.append("a widely-forwarded low-content chain message")
    if f.group_noise_level > 0.6 and value < 0.3:
        risk += 0.15
        reasons.append("from a high-volume, low-signal group")
    if f.user_dismiss_rate > 0.6 and value < 0.3:
        risk += 0.1
        reasons.append("the user has been dismissing most recent notifications")
    if f.quiet_hours and value < 0.5:
        risk += 0.2
        reasons.append("arrives during the user's quiet hours")

    net = value - risk

    if risk >= 0.55 and value < 0.5:
        action = "mute"
    elif net >= 0.35 or (f.direct_mention and value >= 0.4) or (message_type == "urgent" and value >= 0.3):
        action = "notify"
    elif net <= -0.1 or (risk > 0.25 and value < 0.2):
        action = "mute"
    else:
        action = "digest"

    # A message with literally no useful signal at all and no history -> digest, not mute,
    # unless it already tripped a risk condition above (kept conservative on unknowns).
    if message_type == "unknown" and action == "mute" and risk < 0.3:
        action = "digest"

    if not reasons:
        reasons.append("no strong urgency, risk, or repetition signal found; treated as routine content")

    confidence = 0.6 + min(0.3, abs(net) * 0.4) + (0.05 if message_type != "unknown" else -0.05)
    confidence = max(0.55, min(0.93, confidence))

    reason_text = _compose_reason(action, message_type, reasons)
    return Verdict(action=action, message_type=message_type, confidence=confidence, reason=reason_text, tags=tags)


def _classify_type(f: Features, scam_advisory: bool) -> str:
    conv = f.message.get("conversation_type")

    # A widely-forwarded chain message is only classified as "forward"/
    # "greeting" when it isn't *also* clearly promotional/transactional/
    # event content - a marketing blast that happens to be bulk-sent is
    # still a promotion, not a generic forward.
    if f.forward_signal and (
        f.greeting_signal
        or not (f.payment_signal or f.event_signal or f.urgency_signal or f.promo_signal)
    ):
        return "greeting" if f.greeting_signal else "forward"
    if f.greeting_signal and not f.urgency_signal:
        return "greeting"
    if f.promo_signal and not f.payment_signal:
        return "promotion"
    # A verified business warning users *about* scams (e.g. "we will never
    # ask for your OTP") is a safety advisory, not a payment/business update -
    # check this before the payment-keyword match below so words like
    # "payment details" inside the advisory don't get mis-typed.
    if scam_advisory:
        return "business_update"
    if f.payment_signal and (f.business is not None or conv == "business"):
        return "payment" if re.search(r"(payment|invoice|bill|emi|due|autopay|transaction)", f.combined_text, re.IGNORECASE) else "business_update"
    # A casual @-mention asking for a call/response inside a group chat is a
    # personal ask, even if it happens to mention a pickup/timing detail -
    # only a *formal* administrative notice (circular/RSVP/consent form)
    # should be typed as an "event" once a direct mention is present.
    if f.direct_mention and conv == "group" and not f.admin_event_signal and not f.urgency_signal:
        return "personal"
    if conv == "group" and (f.group or {}).get("group_type") == "marketplace" and not f.admin_event_signal:
        return "promotion"
    if f.event_signal:
        return "event"
    # Only genuine time-pressure makes something "urgent". A direct @-mention on
    # its own still boosts the *action* toward notify (see the scoring below),
    # but a casual "@user can you call when free?" is a personal message.
    if f.urgency_signal:
        return "urgent"
    if conv == "personal":
        # A first/unfamiliar contact with no engagement history is better
        # described as unknown than as an established personal conversation.
        if f.sender_familiarity <= 0.0 and not f.payment_signal:
            return "unknown"
        return "personal"
    if f.direct_mention:
        return "personal"
    if f.business is not None:
        return "business_update"
    if conv == "group" and not f.combined_text.strip():
        return "unknown"
    return "personal" if conv == "group" else "unknown"


def _compose_reason(action: str, message_type: str, reasons: List[str]) -> str:
    top = reasons[:2]
    joined = " and ".join(top) if len(top) <= 2 else "; ".join(top[:2])
    verb = {"notify": "interrupt the user now", "digest": "wait for a digest", "mute": "be suppressed"}[action]
    return f"Classified as {message_type}; {joined}, so the message should {verb}."
