#!/usr/bin/env python3
"""CLI: Normalize a PRD into a canonical Spec JSON."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repodesign.spec_normalizer.normalize import normalize_spec, normalize_spec_manual


def main():
    parser = argparse.ArgumentParser(description="Normalize a PRD into Spec JSON")
    parser.add_argument("input", help="Path to PRD text file, or '-' for stdin")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--spec-id", default="spec-001", help="Spec ID")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Read input
    if args.input == "-":
        raw_prd = sys.stdin.read()
    else:
        raw_prd = Path(args.input).read_text()

    spec = normalize_spec(raw_prd, spec_id=args.spec_id, provider=args.provider)

    output_json = spec.model_dump_json(indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json)
        print(f"Saved to: {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
