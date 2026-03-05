#!/usr/bin/env python3
"""Fetch a hand-picked set of repos with real code structure and language diversity.

These repos have: API routes, database models, infrastructure, CI/CD, Docker.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from repodesign.curation.scrape_repos import GitHubScraper, save_repo_list
from repodesign.curation.classify_scale import classify_all

# Hand-picked repos with real code structure + language diversity
TEST_REPOS = [
    # Python — FastAPI + SQLModel + PostgreSQL + Docker + CI
    "tiangolo/full-stack-fastapi-template",
    # Go — REST API + SQLite + auth + realtime subscriptions
    "pocketbase/pocketbase",
    # TypeScript — Next.js + Prisma ORM + tRPC + Docker
    "formbricks/formbricks",
    # Rust — Rails-like web framework with MVC, DB migrations, API
    "loco-rs/loco",
    # Java — Spring Boot CMS + JPA + REST API + Docker
    "halo-dev/halo",
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output = "data/repo_list_test.json"

    scraper = GitHubScraper()
    entries = scraper.fetch_repos_by_name(TEST_REPOS, enrich=True)

    # Classify scale tiers
    classify_all(entries)

    save_repo_list(entries, output)

    print(f"\nFetched {len(entries)} repos:")
    for e in entries:
        print(f"  {e.full_name:40s} {e.primary_language:12s} {e.star_count:>8,} stars  "
              f"contributors={e.num_contributors}  CI={e.has_ci}  Docker={e.has_docker}  "
              f"langs={list(e.languages.keys())[:5]}")
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
