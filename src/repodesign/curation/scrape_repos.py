"""GitHub repository scraping: find and collect repos for the training dataset."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


@dataclass
class RepoEntry:
    """A single repository entry in the curated dataset."""

    name: str
    full_name: str
    url: str
    clone_url: str
    description: str
    primary_language: str
    languages: dict[str, int] = field(default_factory=dict)
    star_count: int = 0
    fork_count: int = 0
    num_contributors: int = 0
    size_kb: int = 0
    topics: list[str] = field(default_factory=list)
    has_ci: bool = False
    has_docker: bool = False
    has_wiki: bool = False
    open_issues: int = 0
    license: str | None = None
    scale_tier: str | None = None


class GitHubScraper:
    """Scrape GitHub for well-structured repositories."""

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self._request_count = 0

    def _get(self, url: str, params: dict | None = None) -> dict | list:
        """Make a rate-limit-aware GET request."""
        self._request_count += 1
        resp = self.session.get(url, params=params)

        # Handle rate limiting
        if resp.status_code == 403:
            reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_time - time.time(), 60)
            logger.warning(f"Rate limited. Waiting {wait:.0f}s...")
            time.sleep(wait)
            resp = self.session.get(url, params=params)

        resp.raise_for_status()
        return resp.json()

    def search_repos(
        self,
        language: str,
        min_stars: int = 50,
        min_size: int = 100,
        max_size: int = 50000,
        max_results: int = 100,
    ) -> list[RepoEntry]:
        """Search GitHub for repos matching criteria."""
        query = f"language:{language} stars:>={min_stars} size:{min_size}..{max_size}"
        entries = []
        page = 1
        per_page = min(100, max_results)

        while len(entries) < max_results:
            logger.info(f"Searching: {query} (page {page})")
            data = self._get(
                f"{GITHUB_API}/search/repositories",
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": per_page,
                    "page": page,
                },
            )

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                if len(entries) >= max_results:
                    break
                entry = RepoEntry(
                    name=item["name"],
                    full_name=item["full_name"],
                    url=item["html_url"],
                    clone_url=item["clone_url"],
                    description=item.get("description") or "",
                    primary_language=item.get("language") or language,
                    star_count=item.get("stargazers_count", 0),
                    fork_count=item.get("forks_count", 0),
                    size_kb=item.get("size", 0),
                    topics=item.get("topics", []),
                    has_wiki=item.get("has_wiki", False),
                    open_issues=item.get("open_issues_count", 0),
                    license=item.get("license", {}).get("spdx_id") if item.get("license") else None,
                )
                entries.append(entry)

            page += 1
            time.sleep(1)  # Be nice to the API

        return entries

    def enrich_repo(self, entry: RepoEntry) -> RepoEntry:
        """Enrich a repo entry with contributor count and CI/Docker detection."""
        # Get contributor count
        try:
            contributors = self._get(
                f"{GITHUB_API}/repos/{entry.full_name}/contributors",
                params={"per_page": 1, "anon": "true"},
            )
            # GitHub returns a list; header has total count
            entry.num_contributors = len(contributors) if isinstance(contributors, list) else 0
        except Exception:
            pass

        # Check for CI and Docker files
        try:
            contents = self._get(f"{GITHUB_API}/repos/{entry.full_name}/contents")
            if isinstance(contents, list):
                names = {c["name"] for c in contents}
                entry.has_docker = "Dockerfile" in names or "docker-compose.yml" in names
                entry.has_ci = ".github" in names or ".gitlab-ci.yml" in names or ".travis.yml" in names
        except Exception:
            pass

        # Get languages breakdown
        try:
            entry.languages = self._get(f"{GITHUB_API}/repos/{entry.full_name}/languages")
        except Exception:
            pass

        time.sleep(0.5)
        return entry

    def scrape_dataset(
        self,
        languages: list[str] | None = None,
        repos_per_language: int = 70,
        enrich: bool = True,
    ) -> list[RepoEntry]:
        """Scrape a full dataset of repos across languages.

        Default targets ~200 repos: 70 Python + 70 JavaScript + 30 Go + 30 TypeScript.
        """
        if languages is None:
            languages = ["python", "javascript", "go", "typescript"]

        all_entries: list[RepoEntry] = []
        for lang in languages:
            count = repos_per_language if lang in ("python", "javascript") else repos_per_language // 2
            logger.info(f"Scraping {count} {lang} repos...")
            entries = self.search_repos(language=lang, max_results=count)
            if enrich:
                for i, entry in enumerate(entries):
                    logger.info(f"  Enriching {i + 1}/{len(entries)}: {entry.full_name}")
                    self.enrich_repo(entry)
            all_entries.extend(entries)

        logger.info(f"Total repos scraped: {len(all_entries)}")
        return all_entries


def save_repo_list(entries: list[RepoEntry], output_path: str) -> None:
    """Save repo list to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)
    logger.info(f"Saved {len(entries)} repos to {output_path}")


def load_repo_list(input_path: str) -> list[RepoEntry]:
    """Load repo list from JSON."""
    with open(input_path) as f:
        data = json.load(f)
    return [RepoEntry(**d) for d in data]
