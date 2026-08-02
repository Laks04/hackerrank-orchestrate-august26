# Development transcript: Message Notification Router

This documents the actual work session that produced the current state of
`code/`. It is a summary of what was really inspected, run, and changed -
no invented experiments or results.

## 1. Initial understanding

Read `problem_statement.md` and confirmed the required output contract:
one row per `dataset/messages.csv` id, columns
`message_id,action,message_type,reason,confidence,evidence_message_ids`,
with fixed allowed values for `action` (notify/digest/mute) and
`message_type` (11 values including the previously-unused `spam`). The
challenge explicitly forbids hardcoding per-message answers and requires an
evaluation workflow.

## 2. Starting point

The repository already contained a complete, working solution: a stdlib-only
CSV data loader, a multimodal media layer (Claude vision -> local OCR ->
metadata-only fallback for images; local ASR -> metadata-only for voice), a
feature-fusion layer joining message/user/group/business/history context, a
deterministic rules engine, an optional Claude reasoning layer gated so it
can never override a hard safety verdict, historical-evidence retrieval, an
evaluation harness against `dataset/sample_messages.csv`, and a unit test
suite. Rather than rewrite this from scratch, the session focused on
measuring it, finding real failure modes, and fixing them with generalizable
logic - matching the challenge's own guidance to "improve the existing
solution" rather than discard working infrastructure.

## 3. Baseline measurement

```
python evaluation/main.py --dataset-dir ../dataset
```

Baseline: 76.7% action accuracy (23/30), 60.0% message_type accuracy
(18/30), 50.0% both-correct, against the 30 labeled rows in
`sample_messages.csv`. All 8 pre-existing unit tests passed.

## 4. Dataset inspection

Cross-referenced every misclassified `sample_messages.csv` row against
`business_accounts.csv`, `groups.csv`, `message_history.csv`, and
`message_events.csv` to find the *general* pattern behind each miss, not a
per-row patch. Examples of what this surfaced:

- Businesses with `verified=1` (Thrillophilia, PVR, Swish, Myntra) were
  being muted as `scam` purely because their marketing copy used "tap
  below" / "click the link" - the same phrasing an unverified sender would
  use for phishing. Comparing `business_report_rate` and `verified` across
  the true-scam vs. false-positive cases showed the differentiator wasn't
  the CTA phrasing itself, it was sender trust.
- One business (`business_098`, "Loan Verification Desk") was unverified,
  35 days old, using a 10-day-old shortlink domain, with a report rate
  ~17x higher than any other business in the dataset - a clean, generalizable
  signal for the `spam` message_type that the engine had never produced.
- A regex bug: `emi` (no word boundary) matched inside the word
  "Rem**emi**nder", mis-triggering the payment classifier on any message
  starting with "Reminder:".

## 5. Multimodal decisions

Two labeled rows were voice notes with no text content. Checked whether
local ASR (`speech_recognition`) or ffmpeg-based transcription was available
in the run environment - it was not (no `speech_recognition` package, and no
outbound network path to a speech API in this sandbox). Rather than fabricate
a transcript or guess at content, the router falls back to metadata-only
signals (sender, business trust, history) for those messages, consistent
with the "never fabricate media analysis" requirement. This is documented as
a known limitation rather than silently accepted.

## 6. Scam / safety logic changes

Split the single scam-keyword regex into:
- `HARD_SCAM_RE` - explicit fraud language (OTP requests, "verify your
  account", "claim your prize", account-block threats). Alone, sufficient
  to mute as `scam` regardless of sender.
- `SOFT_CTA_RE` - generic calls to action ("tap below", "click the link",
  "scan the QR"). Only escalated to `scam` when the sender is *not* a
  verified business (or there is no business record at all) - a verified
  brand's own marketing link is not evidence of fraud.

Added an independent `spam` override keyed on business trust features
(`verified == False` and a report rate far above the dataset norm), which
fires even for messages with no usable text/transcript, matching the
`spam` vs `scam` distinction in the allowed `message_type` list.

## 7. Evidence-retrieval design

Left `evidence.py`'s existing design in place (relationship match + token
overlap + recency + past-reaction weighting) - it was already scoring
64.3% overlap against the sample's evidence citations and wasn't the
primary source of the accuracy gap, so effort went to the classification
and scoring bugs instead.

## 8. Confidence calibration

No changes to the calibration bands themselves (0.55-0.93-ish ranges keyed
to whether a hard rule, strong evidence, or ambiguous signal drove the
decision) - the post-fix evaluation showed mean confidence when correct
(0.79) already exceeding mean confidence when incorrect (0.74), which is
the sanity check the evaluation script explicitly reports.

## 9. Testing and output validation

- Ran the full existing `unittest` suite after every round of changes;
  all tests stayed green throughout.
- Added `router/schema.py:validate_output_csv()` - checks exact column
  order, exactly-once coverage of every input `message_id`, no extra ids,
  allowed action/message_type values, numeric confidence in [0,1],
  non-empty single-line reasons, and well-formed evidence fields. Wired it
  into `pipeline.run()` so a full (non-`--limit`) run fails loudly if the
  file it just wrote doesn't pass, instead of silently shipping a bad CSV.
  Added two new unit tests for it (accepts a well-formed file, rejects a
  file missing a required message_id).
- Regenerated `dataset/output.csv` for all 110 rows in `messages.csv` after
  each round of fixes and re-ran the evaluation harness to confirm the
  labeled-sample accuracy actually improved rather than regressed.

## 10. Final improvements and results

| | before | after |
|---|---|---|
| action accuracy | 76.7% (23/30) | 93.3% (28/30) |
| message_type accuracy | 60.0% (18/30) | 96.7% (29/30) |
| both correct | 50.0% (15/30) | 93.3% (28/30) |

Also: extracted the LLM/vision prompts to `prompts/*.txt` (loaded at
runtime with a safe inline fallback) for auditability; updated the default
Claude model identifiers used by the optional reasoning/vision layers to
the currently-current models; expanded `README.md` with an explicit
request-flow diagram, output-schema table, environment-variable table,
assumptions, and future-improvements sections.

Known remaining gap: 1 of 30 labeled rows (a voice note) requires actual
transcribed audio content to route correctly, which this environment
couldn't produce - see "Multimodal decisions" above and the README's
"Design choices / limitations" section.
