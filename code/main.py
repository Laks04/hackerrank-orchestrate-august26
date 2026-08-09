#!/usr/bin/env python3
"""CLI entrypoint for the Message Notification Router.

Usage:
    python main.py --dataset-dir ../dataset --output-path ../dataset/output.csv
    python main.py --dataset-dir ../dataset --no-llm   # force pure rules engine
    python main.py --dataset-dir ../dataset --limit 10 --verbose

Reads dataset/messages.csv plus every context CSV, routes each message
through router/pipeline.py, and writes output.csv with the exact required
schema: message_id,action,message_type,reason,confidence,evidence_message_ids

Secrets: reads ANTHROPIC_API_KEY from the environment only (never hardcode
it). If it isn't set, the system automatically runs in pure deterministic
mode - see router/rules.py.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router.pipeline import run  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Message Notification Router")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"),
        help="Path to the dataset/ folder (default: ../dataset relative to this file)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to write predictions (default: <dataset-dir>/output.csv)",
    )
    parser.add_argument("--no-llm", action="store_true", help="Disable the optional Claude reasoning layer; pure rules engine.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N messages (useful for quick smoke tests).")
    parser.add_argument("--verbose", action="store_true", help="Print per-message progress and fallback warnings.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_dir = os.path.abspath(args.dataset_dir)
    output_path = os.path.abspath(args.output_path) if args.output_path else os.path.join(dataset_dir, "output.csv")

    try:
        # Optional: load a .env file next to this script if python-dotenv is available.
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        except Exception:
            pass

        results = run(
            dataset_dir=dataset_dir,
            output_path=output_path,
            use_llm=not args.no_llm,
            verbose=args.verbose,
            limit=args.limit,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: output.csv failed final validation: {exc}", file=sys.stderr)
        return 1

    llm_used = sum(1 for r in results if r.used_llm)
    counts = {}
    for r in results:
        counts[r.action] = counts.get(r.action, 0) + 1
    print(f"Routed {len(results)} messages -> {output_path}")
    print(f"Action breakdown: {counts}")
    print(f"LLM-refined: {llm_used}/{len(results)} (rest used the deterministic rules engine)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
