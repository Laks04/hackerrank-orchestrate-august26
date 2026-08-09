<div align="center">

# 📬 Message Notification Router

**A personalized AI system that decides which WhatsApp messages deserve your attention right now — and which can wait, or disappear.**

Built for the **HackerRank Orchestrate** hackathon.

![Action Accuracy](https://img.shields.io/badge/action%20accuracy-93.3%25-brightgreen)
![Type Accuracy](https://img.shields.io/badge/message__type%20accuracy-96.7%25-brightgreen)
![Tests](https://img.shields.io/badge/tests-10%2F10%20passing-brightgreen)
![Dependencies](https://img.shields.io/badge/dependencies-zero%20required-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

</div>

---

## 🧠 The problem

WhatsApp is noisy: family chats, society notices, school updates,
coworker pings, business promos, poster images, voice notes, and the
occasional scam — all in one stream. Treat every message the same and two
things go wrong: important messages get buried, and junk keeps
interrupting you.

This router reads every incoming message — text, image poster, or voice
note — and decides, **personalized to the receiving user**:

| Action | Meaning |
|:---:|---|
| 🔔 `notify` | Important enough to interrupt now |
| 📥 `digest` | Useful, can wait and be shown later |
| 🔇 `mute` | Low-value, repetitive, unwanted, or unsafe |

A sale poster might be gold for one user and noise for another. A payment
reminder is fine from a trusted admin, risky from a stranger. A muted
family group can still have an urgent @mention worth surfacing. And a
clear scam gets muted no matter how engaged the user usually is.

---

## 📊 Results

Evaluated against the 30 labeled rows in `dataset/sample_messages.csv`:

<div align="center">

| Metric | Score |
|:---|:---:|
| ✅ Action accuracy | **93.3%** (28/30) |
| ✅ Message-type accuracy | **96.7%** (29/30) |
| ✅ Both correct | **93.3%** (28/30) |
| 🧪 Unit tests | **10/10 passing** |

</div>

All **110 messages** in `dataset/messages.csv` are routed in
[`dataset/output.csv`](./dataset/output.csv), fully schema-validated.

---

## 🗺️ Where things live

| What | Where |
|---|---|
| 📘 Full architecture, flow diagram, safety design, setup | [`code/README.md`](./code/README.md) |
| 🧩 Source code | [`code/router/`](./code/router/) |
| 📏 Evaluation harness | [`code/evaluation/main.py`](./code/evaluation/main.py) |
| ✅ Tests | [`code/tests/`](./code/tests/) |
| 📝 Development-process writeup | [`code/chat_transcript.md`](./code/chat_transcript.md) |
| 💬 Prompts (optional LLM/vision layer) | [`code/prompts/`](./code/prompts/) |
| 📤 Final predictions | [`dataset/output.csv`](./dataset/output.csv) |
| 📋 Original challenge spec | [`problem_statement.md`](./problem_statement.md) |

---

## 🚀 Quick start

```bash
cd code
python3 -m unittest discover -s tests -v            # run tests
python3 main.py --dataset-dir ../dataset --output-path ../dataset/output.csv --verbose
python3 evaluation/main.py --dataset-dir ../dataset  # accuracy report
```

Runs on the **Python standard library alone** — zero installation
required. Optional upgrades (Claude vision/reasoning, local OCR, local
voice transcription) and their environment variables are documented in
[`code/README.md`](./code/README.md).

---

## ⚙️ How it works


```
 dataset/*.csv
       │
       ▼
 1. data_loader.py      parse + index every provided CSV
       │
       ▼
 2. media.py             image/voice note? → Claude vision → local
       │                 OCR/ASR → metadata-only (never crashes)
       ▼
 3. features.py          fuse message + user + group + business +
       │                 history + media into one signal object
       ▼
 4. rules.py             deterministic safety-net scoring:
       │                   a) hard safety overrides (scam / spam)
       │                   b) message_type classification
       │                   c) value vs. risk → notify / digest / mute
       ▼
 5. llm.py (optional)    only if ANTHROPIC_API_KEY is set — Claude
       │                 can refine ambiguous cases, never overrides
       │                 a hard safety mute
       ▼
 6. evidence.py +        cite relevant historical messages, then
    schema.py            validate the final CSV against the schema
       │
       ▼
 dataset/output.csv
```

A **deterministic rules engine** does the heavy lifting and works with
zero dependencies or API keys. Hard safety rules (scam language, prompt
injection, high-risk unverified senders) always take priority over
personalization — a muted family group can never suppress an actual
emergency, and a clean scam is muted regardless of how much a user usually
engages with that sender. An **optional Claude reasoning layer** can add
nuance on ambiguous cases when an API key is configured, but by design it
can never overrule a hard safety mute.

Full design rationale, assumptions, and limitations are in
[`code/README.md`](./code/README.md).

</details>

---

<div align="center">

## 📋 About this repository

</div>

This started from the HackerRank Orchestrate starter repo for the
**Message Notification Router** challenge — build a system that decides
which WhatsApp messages deserve immediate attention, which can wait, and
which should be muted, personalized per user and safe against scams and
manipulation. See [`problem_statement.md`](./problem_statement.md) for the
full original task spec, schema, and submission requirements.
