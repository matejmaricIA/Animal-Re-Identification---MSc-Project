#!/usr/bin/env python3
"""Create a metadata-only development protocol; no feature extraction."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from reid_demo.benchmark import file_digest, write_json
from reid_demo.benchmark_protocol import build_protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["time_open"], default="time_open")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fitting-fraction", type=float, default=0.6)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--unknown-fraction", type=float, default=0.2)
    parser.add_argument("--query-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.metadata, dtype={"image_id": str, "identity": str})
    plan = build_protocol(frame, file_digest(args.metadata), seed=args.seed, split=args.split,
                          fitting_fraction=args.fitting_fraction,
                          calibration_fraction=args.calibration_fraction,
                          unknown_fraction=args.unknown_fraction, query_fraction=args.query_fraction)
    output = Path(args.out)
    if output.exists():
        import json
        if json.loads(output.read_text()) != plan:
            raise ValueError("Output contains a different protocol; choose a new --out.")
    write_json(output, plan)
    for role, pool in plan["pools"].items():
        inventory = pool["inventory"]
        print(f"{role}: {inventory['images']} images, {inventory['identities']} identities")
        if role != "fitting":
            print(f"  gallery={len(pool['gallery_ids'])}, query={len(pool['query_ids'])}, "
                  f"known={pool['known_queries']}, unknown={pool['unknown_queries']}, "
                  f"excluded={len(pool['excluded'])}")
    print(f"Protocol: {output}\nFingerprint: {plan['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

