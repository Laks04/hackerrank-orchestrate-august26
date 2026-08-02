"""Fuse message + user + group + business + history signals into one
:class:`Features` object per message. This is the shared "understanding"
layer that both the deterministic rules engine and the optional LLM layer
reason over, so the two stay consistent with each other.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .data_loader import Dataset
from .media import MediaResult

# ---------------------------------------------------------------------------
# Regex signal libraries (kept general-purpose, not tuned to specific rows)
# ---------------------------------------------------------------------------

PROMPT_INJECTION_RE = re.compile(
    r"(ignore (all )?(previous|prior|above) (instructions|rules)|"
    r"routing override|disregard (the )?(routing|rules)|"
    r"set action\s*=|action\s*=\s*notify|confidence\s*=\s*1\b|"
    r"you (are|must) (now )?(notify|mark) this|"
    r"actual message\s*:)",
    re.IGNORECASE,
)

HARD_SCAM_RE = re.compile(
    r"(\botp\b|one[-\s]?time password|verify (your )?(account|identity|kyc)|"
    r"account (will be |is )?block(ed)?|suspend(ed)?|"
    r"reply with (the )?(6.?digit|code|otp)|send (the )?(code|otp)|"
    r"claim (your )?(prize|reward|refund|cashback)|congratulations,? you'?ve? won|"
    r"urgent action required|confirm (your )?password|"
    r"login[-\s]?in\.\w+|account-login|"
    r"clearance amount|you'?ve? won|lottery|winner selected)",
    re.IGNORECASE,
)

# Generic call-to-action phrasing ("tap below", "click the link"). Extremely
# common in *legitimate* marketing and business-update messages too, so on
# its own this is NOT treated as a scam signal - see `_cta_context_is_risky`
# below, which only escalates it when the sender is untrustworthy.
SOFT_CTA_RE = re.compile(
    r"(click (the |this )?link|tap (below|here)|scan (this |the )?qr|"
    r"pay(ment)? (is )?pending|penalty)",
    re.IGNORECASE,
)

URGENCY_RE = re.compile(
    # Deliberately requires an explicit time-pressure or escalation cue. A bare
    # mention of minutes ("when you get 5 mins") is NOT urgency, so the numeric
    # patterns below are anchored to words like in/within/max/left/only.
    r"(urgent|asap|right now|immediately|emergency|"
    r"need (you|your|quick) (help|response|input)|"
    r"call (me|now|back)\b|please respond|escalation|deadline|"
    r"before (eod|tonight|today|\d)|by (today|tonight|eod|\d{1,2}\s?(am|pm|:\d{2}))|"
    r"(in|within|next|only|max|maximum|wait)\s?\d+\s?(min|mins|minute|minutes|hour|hours)|"
    r"\d+\s?(min|mins|minutes)\s?(max|left|only)|"
    r"heads[-\s]?up|can(not|'?t) wait|last[-\s]?minute|"
    r"(leaving|starting|closing) (early|in \d+)|"
    r"pulled (to|forward)|same[-\s]day)",
    re.IGNORECASE,
)

GREETING_RE = re.compile(
    r"(good morning|good night|gm\b|have a (nice|great|blessed) day|"
    r"stay positive|god bless|bhagwan|positive energy|share (this )?(blessing|positivity))",
    re.IGNORECASE,
)

FORWARD_HINT_RE = re.compile(
    r"(forward(ed)? as received|fwd\b|forward this to|share (this )?with|"
    r"send (this )?to (\d+|all|everyone)|pls forward|please forward)",
    re.IGNORECASE,
)

PROMO_RE = re.compile(
    r"(% ?off|\bsale\b|discount|\boffer\b|coupon|promo\s?code|use code|flat \d+%|"
    r"limited time|hurry|shop now|buy now|new collection|cashback|deal of the day|"
    r"reply stop to unsubscribe|"
    r"selling |for sale|slightly used|dm if interested|swap for|pickup near main gate|"
    r"no crash damage|not using it anymore)",
    re.IGNORECASE,
)

# Formal/administrative event language (circulars, RSVPs, consent forms). Used
# to keep a casual "@mention, can you call?" from being mis-typed as an
# official event notice just because it happens to mention a pickup time.
ADMIN_EVENT_RE = re.compile(
    r"(circular|rsvp|consent (note|form)|parent(-|\s)?teacher|form is open|"
    r"\bnotice\b|holiday (list|notice)|school (circular|notice))",
    re.IGNORECASE,
)

PAYMENT_RE = re.compile(
    r"(payment|invoice|bill(ed)?|due (amount|date)|\bemi\b|autopay|"
    r"card (ending|update)|account (statement|balance)|transaction|refund processed|"
    r"order (ending|#)|delivery|shipment|dispatched|out for delivery)",
    re.IGNORECASE,
)

# Time-bound / action-required phrasing that separates a genuinely
# now-relevant business update (package arriving today, appointment before
# a specific time) from a passive one (a feedback request, a thank-you note)
# even when both otherwise match the same business relationship.
ACTIONABLE_RE = re.compile(
    r"(\btoday\b|\btonight\b|before the scheduled time|expected to (reach|arrive)|"
    r"arriving (today|soon)|out for delivery|ready for (review|pickup|collection)|"
    r"expires (today|tonight|soon)|final stage|last (day|chance))",
    re.IGNORECASE,
)

EVENT_RE = re.compile(
    r"(circular|timing|schedule|reminder|rsvp|meeting|appointment|pickup|"
    r"bus (is|will|leaving)|class (starts|timing)|event|form is open|consent note|"
    r"parent(-|\s)?teacher|holiday|notice)",
    re.IGNORECASE,
)

URGENCY_NEGATION_RE = re.compile(
    r"(nothing urgent|not urgent|no rush|don'?t call|do not call|no need to (call|rush)|"
    r"whenever (you'?re|you are|is) (free|convenient)|talk (tomorrow|later)|no hurry)",
    re.IGNORECASE,
)

MENTION_RE = re.compile(r"@(u_\w+)")


@dataclass
class Features:
    message: Dict[str, str]
    user: Optional[Dict[str, str]]
    group: Optional[Dict[str, str]]
    group_member: Optional[Dict[str, str]]
    business: Optional[Dict[str, str]]
    user_business: Optional[Dict[str, str]]
    media: Optional[MediaResult]

    combined_text: str = ""
    direct_mention: bool = False
    quiet_hours: bool = False
    prompt_injection: bool = False
    scam_keywords: bool = False
    hard_scam_signal: bool = False
    soft_cta_signal: bool = False
    urgency_signal: bool = False
    greeting_signal: bool = False
    forward_signal: bool = False
    promo_signal: bool = False
    payment_signal: bool = False
    event_signal: bool = False
    admin_event_signal: bool = False
    actionable_signal: bool = False

    group_muted: bool = False
    is_admin_sender: bool = False
    group_noise_level: float = 0.0  # 0-1, higher = noisier group

    business_verified: bool = False
    domain_mismatch: bool = False
    young_domain: bool = False
    business_report_rate: float = 0.0
    user_opted_out_promo: bool = False
    user_has_relevant_business_history: bool = False

    sender_familiarity: float = 0.0  # 0-1 based on prior reply/open behaviour w/ this sender
    repetition_penalty: float = 0.0  # 0-1, higher = user has ignored/muted/reported similar before
    user_dismiss_rate: float = 0.0  # 0-1 over last 30d, general "notification fatigue"
    user_report_rate: float = 0.0

    forwarded_count: int = 0
    notes: List[str] = field(default_factory=list)


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_hhmm(value: str):
    try:
        h, m = value.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _in_quiet_hours(dnd_window: str, created_at: str) -> bool:
    if not dnd_window or "-" not in dnd_window or not created_at:
        return False
    try:
        start_s, end_s = dnd_window.split("-")
        start, end = _parse_hhmm(start_s), _parse_hhmm(end_s)
        ts = datetime.strptime(created_at.strip(), "%Y-%m-%d %H:%M")
        minute_of_day = ts.hour * 60 + ts.minute
    except Exception:
        return False
    if start is None or end is None:
        return False
    if start <= end:
        return start <= minute_of_day <= end
    return minute_of_day >= start or minute_of_day <= end  # window wraps midnight


def build_features(ds: Dataset, message: Dict[str, str], media: Optional[MediaResult]) -> Features:
    user_id = message.get("user_id")
    group_id = message.get("group_id") or None
    business_id = message.get("business_id") or None
    sender_user_id = message.get("sender_user_id") or None

    user = ds.get_user(user_id)
    group = ds.get_group(group_id)
    group_member = ds.get_group_member(group_id, user_id)  # the RECEIVING user's membership row
    business = ds.get_business(business_id)
    user_business = ds.get_user_business_history(user_id, business_id)

    text = message.get("message_text") or ""
    media_text = (media.text if media else "") or ""
    media_caption = (media.caption if media else "") or ""
    combined_text = " \n".join(t for t in [text, media_text, media_caption] if t)

    feats = Features(
        message=message,
        user=user,
        group=group,
        group_member=group_member,
        business=business,
        user_business=user_business,
        media=media,
        combined_text=combined_text,
    )

    feats.forwarded_count = _safe_int(message.get("forwarded_count"), 0)
    feats.direct_mention = bool(MENTION_RE.search(text)) and (f"@{user_id}" in text)
    feats.quiet_hours = _in_quiet_hours((user or {}).get("do_not_disturb_window", ""), message.get("created_at", ""))

    feats.prompt_injection = bool(PROMPT_INJECTION_RE.search(combined_text))
    feats.hard_scam_signal = bool(HARD_SCAM_RE.search(combined_text))
    feats.soft_cta_signal = bool(SOFT_CTA_RE.search(combined_text))
    urgency_negated = bool(URGENCY_NEGATION_RE.search(combined_text))
    feats.urgency_signal = bool(URGENCY_RE.search(combined_text)) and not urgency_negated
    feats.greeting_signal = bool(GREETING_RE.search(combined_text)) and not feats.hard_scam_signal
    feats.forward_signal = bool(FORWARD_HINT_RE.search(combined_text)) or feats.forwarded_count >= 3
    feats.promo_signal = bool(PROMO_RE.search(combined_text))
    feats.payment_signal = bool(PAYMENT_RE.search(combined_text))
    feats.event_signal = bool(EVENT_RE.search(combined_text))
    feats.admin_event_signal = bool(ADMIN_EVENT_RE.search(combined_text))
    feats.actionable_signal = bool(ACTIONABLE_RE.search(combined_text))

    if media and media.risk_tags:
        if "scam_style" in media.risk_tags or "otp_or_verification" in media.risk_tags:
            feats.hard_scam_signal = True
        if "promotional" in media.risk_tags:
            feats.promo_signal = True

    # --- group signals -----------------------------------------------------
    if group_member is not None:
        feats.group_muted = _safe_int(group_member.get("group_muted_by_user"), 0) == 1
    if group is not None:
        members = _safe_int(group.get("member_count"), 1) or 1
        messages_30d = _safe_int(group.get("messages_30d"), 0)
        # crude noise proxy: messages per member per day, squashed to 0-1
        per_member_per_day = (messages_30d / members) / 30.0
        feats.group_noise_level = min(1.0, per_member_per_day / 2.0)
    if group_id and sender_user_id:
        sender_membership = ds.get_group_member(group_id, sender_user_id)
        if sender_membership is not None:
            feats.is_admin_sender = sender_membership.get("role") == "admin"

    # --- business trust signals --------------------------------------------
    if business is not None:
        feats.business_verified = _safe_int(business.get("verified"), 0) == 1
        official = (business.get("official_domain") or "").strip().lower()
        used = (business.get("domain_used_by_sender") or "").strip().lower()
        feats.domain_mismatch = bool(official) and bool(used) and official != used
        account_age = _safe_int(business.get("account_age_days"), 9999)
        domain_age = _safe_int(business.get("domain_used_by_sender_age_days"), 9999)
        feats.young_domain = domain_age < 90 or (account_age > 0 and domain_age < account_age * 0.1)
        sent = _safe_int(business.get("messages_sent_30d"), 0) or 1
        reports = _safe_int(business.get("user_reports_30d"), 0)
        feats.business_report_rate = min(1.0, reports / sent * 50)  # scaled - reports are rare events

    # A generic "tap below" / "click the link" CTA only becomes a scam signal
    # when the sender itself isn't trustworthy - a verified brand's marketing
    # link is not a scam just because it uses a CTA. An unverified business,
    # or a CTA arriving from something that isn't a business account at all
    # (impersonation), is risky.
    cta_is_risky = feats.business is None or not feats.business_verified
    feats.scam_keywords = feats.hard_scam_signal or (feats.soft_cta_signal and cta_is_risky)

    if user_business is not None:
        feats.user_opted_out_promo = (
            _safe_int(user_business.get("allows_promotions"), 1) == 0
            or bool(user_business.get("promotions_opted_out_at"))
        )
        activity = _safe_int(user_business.get("activity_count_180d"), 0)
        opened = _safe_int(user_business.get("messages_opened_30d"), 0)
        feats.user_has_relevant_business_history = activity > 0 or opened > 0

    # --- historical repetition / familiarity via message_history+events ----
    history_rows: List[Dict[str, str]] = []
    if sender_user_id:
        history_rows.extend(ds.history_by_sender.get(sender_user_id, []))
    if business_id:
        history_rows.extend(ds.history_by_business.get(business_id, []))
    history_rows = [h for h in history_rows if h.get("user_id") == user_id]

    if history_rows:
        opens = replies = dismisses = mutes = reports = 0
        for h in history_rows:
            ev = ds.get_event(h.get("message_id"))
            if not ev:
                continue
            opens += _safe_int(ev.get("message_opened"))
            replies += _safe_int(ev.get("message_replied"))
            dismisses += _safe_int(ev.get("notification_dismissed"))
            mutes += _safe_int(ev.get("muted_after_message"))
            reports += _safe_int(ev.get("message_reported"))
        n = max(1, len(history_rows))
        feats.sender_familiarity = min(1.0, (opens + 2 * replies) / (2 * n))
        feats.repetition_penalty = min(1.0, (dismisses + 2 * mutes + 3 * reports) / (2 * n))

    # --- overall notification fatigue (daily_notification_summary) --------
    daily_rows = ds.daily_summary_by_user.get(user_id, [])
    if daily_rows:
        recent = daily_rows[-7:]
        sent_total = sum(_safe_int(r.get("notifications_sent")) for r in recent) or 1
        dismissed_total = sum(_safe_int(r.get("notifications_dismissed")) for r in recent)
        feats.user_dismiss_rate = min(1.0, dismissed_total / sent_total)

    if user is not None:
        opened_30d = _safe_int(user.get("messages_opened_30d")) or 1
        reported_30d = _safe_int(user.get("messages_reported_30d"))
        feats.user_report_rate = min(1.0, reported_30d / opened_30d * 10)

    return feats
