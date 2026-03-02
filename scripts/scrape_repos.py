#!/usr/bin/env python3
"""CLI: Scrape GitHub repositories for the training dataset."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from repodesign.curation.classify_scale import classify_all
from repodesign.curation.scrape_repos import GitHubScraper, save_repo_list


def main():
    parser = argparse.ArgumentParser(description="Scrape GitHub repos for training dataset")
    parser.add_argument("--languages", nargs="+", default=["python", "javascript", "go", "typescript"])
    parser.add_argument("--per-language", type=int, default=70, help="Repos per language (default 70)")
    parser.add_argument("--output", "-o", default="data/repo_list.json", help="Output path")
    parser.add_argument("--no-enrich", action="store_true", help="Skip enrichment (faster)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    scraper = GitHubScraper()
    entries = scraper.scrape_dataset(
        languages=args.languages,
        repos_per_language=args.per_language,
        enrich=not args.no_enrich,
    )

    # Classify scale tiers
    grouped = classify_all(entries)

    save_repo_list(entries, args.output)

    print(f"\nScraped {len(entries)} repos:")
    for tier, repos in grouped.items():
        print(f"  {tier.value}: {len(repos)}")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
