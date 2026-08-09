#!/usr/bin/env python3
"""Evaluation workflow.

Compares generated predictions against the labeled rows in
dataset/sample_messages.csv (the only ground truth participants are given)
and reports:
  - action accuracy / confusion matrix
  - message_type accuracy / confusion matrix
  - mean confidence on correct vs incorrect predictions (calibration sanity)
  - how often evidence_message_ids overlaps with a labeled example, when both
    provide evidence

This is meant to be run after code/main.py has produced dataset/output.csv,
and is also useful stand-alone: it will (re)generate predictions for just the
labeled subset if --predictions is not supplied, so you can smoke-test the
router without regenerating the full 110-row output.

Usage:
    python evaluation/main.py --dataset-dir ../dataset --predictions ../dataset/output.csv
    python evaluation/main.py --dataset-dir ../dataset   # auto-runs the router on the labeled subset
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from router.data_loader import load_dataset  # noqa: E402
from router.llm import LLMReasoner  # noqa: E402
from router.media import MediaAnalyzer, MediaCache  # noqa: E402
from router.pipeline import route_message  # noqa: E402
from router.schema import INPUT_COLUMNS  # noqa: E402


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_predictions(path: str) -> Dict[str, Dict[str, str]]:
    rows = _read_csv(path)
    return {r["message_id"]: r for r in rows}


def evaluate(dataset_dir: str, predictions: Dict[str, Dict[str, str]]) -> None:
    labeled = _read_csv(os.path.join(dataset_dir, "sample_messages.csv"))
    labeled = [r for r in labeled if r.get("action")]  # only rows with ground truth filled in

    if not labeled:
        print("No labeled rows found in sample_messages.csv - nothing to evaluate.")
        return

    n = len(labeled)
    action_correct = 0
    type_correct = 0
    both_correct = 0
    action_confusion = defaultdict(Counter)  # true_action -> Counter(predicted_action)
    type_confusion = defaultdict(Counter)
    confidences_correct: List[float] = []
    confidences_incorrect: List[float] = []
    evidence_overlap = 0
    evidence_comparable = 0
    missing_predictions = []

    for row in labeled:
        mid = row["message_id"]
        pred = predictions.get(mid)
        if pred is None:
            missing_predictions.append(mid)
            continue

        true_action, pred_action = row["action"], pred.get("action", "")
        true_type, pred_type = row["message_type"], pred.get("message_type", "")

        action_confusion[true_action][pred_action] += 1
        type_confusion[true_type][pred_type] += 1

        a_ok = true_action == pred_action
        t_ok = true_type == pred_type
        action_correct += int(a_ok)
        type_correct += int(t_ok)
        both_correct += int(a_ok and t_ok)

        try:
            conf = float(pred.get("confidence", 0) or 0)
        except ValueError:
            conf = 0.0
        (confidences_correct if a_ok else confidences_incorrect).append(conf)

        true_ev = (row.get("evidence_message_ids") or "none").strip()
        pred_ev = (pred.get("evidence_message_ids") or "none").strip()
        if true_ev != "none":
            evidence_comparable += 1
            true_set = set(true_ev.split(";"))
            pred_set = set(pred_ev.split(";")) if pred_ev != "none" else set()
            if true_set & pred_set:
                evidence_overlap += 1

    print(f"Evaluated {n - len(missing_predictions)}/{n} labeled sample rows")
    if missing_predictions:
        print(f"  WARNING: predictions missing for: {', '.join(missing_predictions)}")

    print(f"\naction accuracy:        {action_correct}/{n} = {action_correct / n:.1%}")
    print(f"message_type accuracy:  {type_correct}/{n} = {type_correct / n:.1%}")
    print(f"both correct:           {both_correct}/{n} = {both_correct / n:.1%}")

    if evidence_comparable:
        print(f"evidence overlap:       {evidence_overlap}/{evidence_comparable} = {evidence_overlap / evidence_comparable:.1%} (rows where ground truth cites evidence)")

    if confidences_correct:
        print(f"\nmean confidence when correct:   {sum(confidences_correct) / len(confidences_correct):.2f}")
    if confidences_incorrect:
        print(f"mean confidence when incorrect: {sum(confidences_incorrect) / len(confidences_incorrect):.2f}  (should be lower than 'correct' if calibration is sane)")

    print("\naction confusion matrix (rows=truth, cols=prediction):")
    actions = sorted({*action_confusion.keys(), *(k for c in action_confusion.values() for k in c)})
    header = "truth\\pred".ljust(12) + "".join(a.ljust(10) for a in actions)
    print(header)
    for truth in actions:
        row_counts = action_confusion.get(truth, Counter())
        print(truth.ljust(12) + "".join(str(row_counts.get(a, 0)).ljust(10) for a in actions))

    print("\nmisclassified rows (truth -> predicted):")
    for row in labeled:
        mid = row["message_id"]
        pred = predictions.get(mid)
        if pred is None:
            continue
        if row["action"] != pred.get("action") or row["message_type"] != pred.get("message_type"):
            print(
                f"  {mid}: truth=({row['action']}, {row['message_type']}) "
                f"pred=({pred.get('action')}, {pred.get('message_type')}) "
                f"reason={pred.get('reason')!r}"
            )


def route_sample_messages(dataset_dir: str, use_llm: bool) -> Dict[str, Dict[str, str]]:
    """sample_messages.csv has its own message_id namespace (sample_msg_*) and is not
    part of messages.csv - it's the only labeled data participants get. We route each
    of its rows through the exact same pipeline (sharing the same context dataset for
    users/groups/business/history joins) and compare against its own label columns.
    """
    ds = load_dataset(dataset_dir)
    cache = MediaCache(os.path.join(dataset_dir, ".media_cache.json"))
    media_analyzer = MediaAnalyzer(use_llm=use_llm, cache=cache)
    llm_reasoner = LLMReasoner() if use_llm else None

    sample_rows = _read_csv(os.path.join(dataset_dir, "sample_messages.csv"))
    predictions: Dict[str, Dict[str, str]] = {}
    for row in sample_rows:
        if not row.get("action"):
            continue  # unlabeled example row, nothing to compare against
        message = {col: row.get(col, "") for col in INPUT_COLUMNS}
        result = route_message(ds, message, media_analyzer, llm_reasoner)
        predictions[result.message_id] = result.to_row()
    cache.save()
    return predictions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate router predictions against dataset/sample_messages.csv")
    default_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset")
    parser.add_argument("--dataset-dir", default=default_dataset)
    parser.add_argument(
        "--predictions",
        default=None,
        help="Path to an existing predictions CSV keyed by message_id (e.g. a run against sample_messages.csv). "
        "If omitted, the router is (re)run directly on the labeled rows in sample_messages.csv.",
    )
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args(argv)

    dataset_dir = os.path.abspath(args.dataset_dir)

    if args.predictions:
        predictions = load_predictions(os.path.abspath(args.predictions))
    else:
        predictions = route_sample_messages(dataset_dir, use_llm=not args.no_llm)

    evaluate(dataset_dir, predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
