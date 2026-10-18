#!/usr/bin/env python3
"""Prepare a model-free CzechLynx audit or import completed manual reviews."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reid_demo.benchmark_input_audit import prepare_audit, summarize_reviews


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--metadata", required=True)
    prepare.add_argument("--protocol-plan", required=True)
    prepare.add_argument("--raw-metadata", required=True)
    prepare.add_argument("--images-root", required=True)
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--sample-size", type=int, default=128)
    prepare.add_argument("--seed", type=int, default=42)
    summary = sub.add_parser("summarize")
    summary.add_argument("--audit-dir", required=True)
    summary.add_argument("--reviews", required=True)
    summary.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            value = prepare_audit(args.metadata, args.protocol_plan, args.raw_metadata, args.images_root,
                                  args.out_dir, sample_size=args.sample_size, seed=args.seed)
            print(f"Audited {len(value['sample'])} crops. Review: {Path(args.out_dir) / 'index.html'}")
            print("Copy reviews_template.csv to a working CSV, fill every row, then use summarize.")
        else:
            value = summarize_reviews(args.audit_dir, args.reviews, args.out_dir)
            print(f"Imported {len(value['sample'])} reviews. Receipt: {Path(args.out_dir) / 'receipt.json'}")
        return 0
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
