#!/usr/bin/env python3
"""CLI: Mine architecture diagrams from repositories."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repodesign.diagrams.mine_diagrams import mine_diagrams_batch, mine_diagrams_from_repo, save_diagram_manifest


def main():
    parser = argparse.ArgumentParser(description="Mine architecture diagrams from repos")
    parser.add_argument("repo_paths", nargs="+", help="Paths to repositories")
    parser.add_argument("--output", "-o", default="data/diagrams/manifest.json", help="Output manifest path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    entries = mine_diagrams_batch(args.repo_paths)

    save_diagram_manifest(entries, args.output)

    # Summary by type
    by_type: dict[str, int] = {}
    for e in entries:
        by_type[e.diagram_type] = by_type.get(e.diagram_type, 0) + 1

    print(f"\nFound {len(entries)} diagrams:")
    for dtype, count in sorted(by_type.items()):
        print(f"  {dtype}: {count}")
    print(f"\nSaved manifest to: {args.output}")


if __name__ == "__main__":
    main()
