#!/usr/bin/env python3
"""CLI: Extract Repo IR from a local repository."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repodesign.extractors.pipeline import extract_repo_ir, save_repo_ir


def main():
    parser = argparse.ArgumentParser(description="Extract Repo IR from a repository")
    parser.add_argument("repo_path", help="Path to the repository")
    parser.add_argument("--url", default="", help="GitHub URL of the repository")
    parser.add_argument("--stars", type=int, default=0, help="Number of stars")
    parser.add_argument("--contributors", type=int, default=0, help="Number of contributors")
    parser.add_argument("--output", "-o", help="Output JSON path (default: data/repo_irs/<name>.json)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM-based summary")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    repo_path = str(Path(args.repo_path).resolve())
    if not Path(repo_path).is_dir():
        print(f"Error: {repo_path} is not a directory")
        sys.exit(1)

    repo_ir = extract_repo_ir(
        repo_path=repo_path,
        repo_url=args.url,
        star_count=args.stars,
        num_contributors=args.contributors,
        skip_llm=args.skip_llm,
        llm_provider=args.provider,
    )

    output_path = args.output or f"data/repo_irs/{Path(repo_path).name}.json"
    save_repo_ir(repo_ir, output_path)

    # Print summary
    print(f"\nRepo IR extracted for: {repo_ir.repo_metadata.name}")
    print(f"  Language: {repo_ir.repo_metadata.primary_language}")
    print(f"  LOC: {repo_ir.repo_metadata.total_loc:,}")
    print(f"  Dependencies: {len(repo_ir.dependencies)}")
    print(f"  API routes: {len(repo_ir.api_routes)}")
    print(f"  Data models: {len(repo_ir.data_models)}")
    print(f"  Scale tier: {repo_ir.repo_metadata.scale_tier}")
    if repo_ir.extraction_warnings:
        print(f"  Warnings: {len(repo_ir.extraction_warnings)}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
