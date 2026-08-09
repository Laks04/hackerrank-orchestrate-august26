"""Load every dataset/*.csv file with the Python standard library only.

Deliberately dependency-free: the grading environment may not have pandas
(or any third-party package) installed, and the project contract requires
the solution to "be runnable from the terminal" without surprises. All
downstream modules consume plain dicts/lists produced here.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


@dataclass
class Dataset:
    """All dataset/*.csv tables plus fast indexes for joining across them."""

    root: str

    messages: List[Dict[str, str]] = field(default_factory=list)
    sample_messages: List[Dict[str, str]] = field(default_factory=list)
    users: List[Dict[str, str]] = field(default_factory=list)
    groups: List[Dict[str, str]] = field(default_factory=list)
    group_members: List[Dict[str, str]] = field(default_factory=list)
    business_accounts: List[Dict[str, str]] = field(default_factory=list)
    user_business_history: List[Dict[str, str]] = field(default_factory=list)
    message_history: List[Dict[str, str]] = field(default_factory=list)
    message_events: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    voice_notes: List[Dict[str, str]] = field(default_factory=list)
    daily_notification_summary: List[Dict[str, str]] = field(default_factory=list)

    # Indexes (built in __post_init__)
    users_by_id: Dict[str, Dict[str, str]] = field(default_factory=dict)
    groups_by_id: Dict[str, Dict[str, str]] = field(default_factory=dict)
    group_member_by_key: Dict[tuple, Dict[str, str]] = field(default_factory=dict)
    group_members_by_group: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    business_by_id: Dict[str, Dict[str, str]] = field(default_factory=dict)
    user_business_by_key: Dict[tuple, Dict[str, str]] = field(default_factory=dict)
    images_by_id: Dict[str, Dict[str, str]] = field(default_factory=dict)
    voice_by_id: Dict[str, Dict[str, str]] = field(default_factory=dict)

    message_events_by_id: Dict[str, Dict[str, str]] = field(default_factory=dict)
    history_by_user: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    history_by_sender: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    history_by_business: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    history_by_group: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: defaultdict(list))
    daily_summary_by_user: Dict[str, List[Dict[str, str]]] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self) -> None:
        for u in self.users:
            self.users_by_id[u["user_id"]] = u
        for g in self.groups:
            self.groups_by_id[g["group_id"]] = g
        for gm in self.group_members:
            self.group_member_by_key[(gm["group_id"], gm["user_id"])] = gm
            self.group_members_by_group[gm["group_id"]].append(gm)
        for b in self.business_accounts:
            self.business_by_id[b["business_id"]] = b
        for ubh in self.user_business_history:
            self.user_business_by_key[(ubh["user_id"], ubh["business_id"])] = ubh
        for img in self.images:
            self.images_by_id[img["image_id"]] = img
        for vn in self.voice_notes:
            self.voice_by_id[vn["voice_note_id"]] = vn
        for ev in self.message_events:
            self.message_events_by_id[ev["message_id"]] = ev
        for h in self.message_history:
            self.history_by_user[h.get("user_id", "")].append(h)
            sender = h.get("sender_user_id") or ""
            if sender:
                self.history_by_sender[sender].append(h)
            biz = h.get("business_id") or ""
            if biz:
                self.history_by_business[biz].append(h)
            grp = h.get("group_id") or ""
            if grp:
                self.history_by_group[grp].append(h)
        for row in self.daily_notification_summary:
            self.daily_summary_by_user[row.get("user_id", "")].append(row)

    # -- convenience lookups -------------------------------------------------
    def get_user(self, user_id: Optional[str]) -> Optional[Dict[str, str]]:
        return self.users_by_id.get(user_id) if user_id else None

    def get_group(self, group_id: Optional[str]) -> Optional[Dict[str, str]]:
        return self.groups_by_id.get(group_id) if group_id else None

    def get_group_member(self, group_id: Optional[str], user_id: Optional[str]) -> Optional[Dict[str, str]]:
        if not group_id or not user_id:
            return None
        return self.group_member_by_key.get((group_id, user_id))

    def get_business(self, business_id: Optional[str]) -> Optional[Dict[str, str]]:
        return self.business_by_id.get(business_id) if business_id else None

    def get_user_business_history(self, user_id: Optional[str], business_id: Optional[str]) -> Optional[Dict[str, str]]:
        if not user_id or not business_id:
            return None
        return self.user_business_by_key.get((user_id, business_id))

    def get_image(self, image_id: Optional[str]) -> Optional[Dict[str, str]]:
        return self.images_by_id.get(image_id) if image_id else None

    def get_voice_note(self, voice_note_id: Optional[str]) -> Optional[Dict[str, str]]:
        return self.voice_by_id.get(voice_note_id) if voice_note_id else None

    def get_event(self, message_id: Optional[str]) -> Optional[Dict[str, str]]:
        return self.message_events_by_id.get(message_id) if message_id else None

    def media_path(self, media_type: str, media_id: str) -> Optional[str]:
        row = None
        if media_type == "image":
            row = self.get_image(media_id)
        elif media_type == "voice":
            row = self.get_voice_note(media_id)
        if not row:
            return None
        rel = row.get("file_path")
        if not rel:
            return None
        return os.path.join(self.root, rel)


def load_dataset(dataset_dir: str) -> Dataset:
    """Read every participant-facing CSV under ``dataset_dir`` into a Dataset."""

    def p(name: str) -> str:
        return os.path.join(dataset_dir, name)

    ds = Dataset(
        root=dataset_dir,
        messages=_read_csv(p("messages.csv")),
        sample_messages=_read_csv(p("sample_messages.csv")),
        users=_read_csv(p("users.csv")),
        groups=_read_csv(p("groups.csv")),
        group_members=_read_csv(p("group_members.csv")),
        business_accounts=_read_csv(p("business_accounts.csv")),
        user_business_history=_read_csv(p("user_business_history.csv")),
        message_history=_read_csv(p("message_history.csv")),
        message_events=_read_csv(p("message_events.csv")),
        images=_read_csv(p("images.csv")),
        voice_notes=_read_csv(p("voice_notes.csv")),
        daily_notification_summary=_read_csv(p("daily_notification_summary.csv")),
    )
    return ds
