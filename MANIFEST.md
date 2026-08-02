# Complete working project - Message Notification Router

This is the full, self-contained project: source code, the complete
dataset (CSVs + media files), a working `dataset/output.csv` already
generated, and every submission deliverable. Verified to run end-to-end
from a completely fresh copy immediately before packaging (see "Verified"
below).

## Layout

```
.
├── README.md              starter repo readme (challenge overview)
├── problem_statement.md   full challenge spec / schema / rules
├── AGENTS.md               AI-coding-tool conventions used during this build
├── log.txt                 session transcript (also inside code.zip's history)
├── code.zip                 pre-zipped code/ - upload as-is for the "Code zip" deliverable
├── code/                     the same source, already unzipped - run it directly from here
│   ├── main.py                 CLI entrypoint
│   ├── router/                  data loading, media, features, rules, LLM layer, pipeline
│   ├── prompts/                  the actual LLM/vision prompts, as text files
│   ├── evaluation/                accuracy/confusion-matrix report vs sample_messages.csv
│   ├── tests/                     unit tests (10/10 passing)
│   ├── README.md                   full architecture + results writeup
│   ├── chat_transcript.md           development-process writeup
│   ├── config.yaml                   documented tunables
│   └── requirements.txt              optional deps (system runs with zero of them installed)
└── dataset/                the full provided dataset, including media/ and the
                              already-generated output.csv (regenerate anytime, see below)
```

## Run it

```bash
cd code
python3 -m unittest discover -s tests -v            # 10/10 should pass
python3 main.py --dataset-dir ../dataset --output-path ../dataset/output.csv --verbose
python3 evaluation/main.py --dataset-dir ../dataset  # accuracy report
```

No installation is required for the above - the router runs on the Python
standard library alone. `pip install -r requirements.txt` inside `code/`
only if you want the *optional* upgrades: Claude vision/reasoning (needs
`ANTHROPIC_API_KEY`, see `code/.env.example`), local image OCR
(`pytesseract` + system `tesseract`), or local voice-note transcription
(`speech_recognition` + `pydub` + system `ffmpeg`).

## Verified (immediately before this zip was built)

Ran from a brand-new copy of this exact folder, not the working copy that
produced it:

```
10/10 unit tests pass
Routed 110/110 messages -> dataset/output.csv
[validate] output.csv passed schema/coverage validation (110 rows)
action accuracy:        28/30 = 93.3%
message_type accuracy:  29/30 = 96.7%
both correct:           28/30 = 93.3%
```

## Submission deliverables (HackerRank)

1. **Code zip** -> upload `code.zip` as-is
2. **Predictions CSV** -> upload `dataset/output.csv` as-is
3. **Chat transcript** -> upload `log.txt` as-is

## Updating your GitHub repo

Copy this zip's `code/` over your repo's `code/` directory, copy
`dataset/output.csv` over your repo's `dataset/output.csv`, commit, and
push. (`dataset/`'s other files are the original challenge data and don't
need to change - they're included here only so the project runs standalone
without needing a separate clone.)
