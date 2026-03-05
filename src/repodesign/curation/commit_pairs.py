"""Commit-pair extraction: find squash-merged PRs and extract before/after data.

Pipeline:
  1. Query GitHub API for merged PRs in a repo
  2. Filter for squash merges (single-commit PRs merged into main)
  3. For each PR, record: before_sha, after_sha (merge commit), diff, PR metadata
  4. Later: clone repo at before_sha, extract RepoIR, feed diff to teacher model

A "commit pair" is:
  - before_sha: the parent of the merge commit (repo state before the feature)
  - after_sha:  the merge commit itself (repo state after the feature)
  - diff:       the changes introduced by the merge commit
  - pr_title / pr_body: the human-written feature description
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

# Heuristics to filter out non-feature PRs
_SKIP_TITLE_PATTERNS = re.compile(
    r"^(bump|update|upgrade|chore|ci|docs|typo|fix typo|refactor|lint|"
    r"dependabot|renovate|release|version|changelog|readme)",
    re.IGNORECASE,
)

# Minimum diff size to be interesting (too small = trivial fix)
MIN_DIFF_FILES = 2
# Maximum diff size to fit in teacher model context
MAX_DIFF_FILES = 50
MAX_DIFF_CHARS = 60_000


@dataclass
class CommitPair:
    """A before/after commit pair from a squash-merged PR."""

    repo_full_name: str
    pr_number: int
    pr_title: str
    pr_body: str
    pr_url: str
    pr_labels: list[str] = field(default_factory=list)
    before_sha: str = ""
    after_sha: str = ""
    merge_commit_sha: str = ""
    diff_text: str = ""
    diff_files: list[str] = field(default_factory=list)
    diff_stats: dict = field(default_factory=dict)


class CommitPairFinder:
    """Find squash-merged PRs and extract commit pairs from a GitHub repo."""

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        self.session.headers["Accept"] = "application/vnd.github.v3+json"

    def _get(self, url: str, params: dict | None = None) -> dict | list:
        """Rate-limit-aware GET request."""
        resp = self.session.get(url, params=params)
        if resp.status_code == 403:
            reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_time - time.time(), 60)
            logger.warning(f"Rate limited. Waiting {wait:.0f}s...")
            time.sleep(wait)
            resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def find_squash_merged_prs(
        self,
        repo_full_name: str,
        max_prs: int = 5,
        max_pages: int = 10,
    ) -> list[CommitPair]:
        """Find squash-merged PRs that look like feature additions.

        Strategy:
          1. List recently merged PRs (sorted by updated)
          2. Filter: must have merge_commit_sha, skip bot/trivial PRs
          3. Check if merge commit has exactly 1 parent (squash merge indicator)
          4. Fetch diff stats to ensure it's a meaningful change

        Args:
            repo_full_name: e.g. "pocketbase/pocketbase"
            max_prs: Max commit pairs to return
            max_pages: Max API pages to scan

        Returns:
            List of CommitPair objects with SHAs and PR metadata filled in.
        """
        pairs: list[CommitPair] = []
        page = 1

        while len(pairs) < max_prs and page <= max_pages:
            logger.info(f"  Scanning PRs page {page} for {repo_full_name}...")
            prs = self._get(
                f"{GITHUB_API}/repos/{repo_full_name}/pulls",
                params={
                    "state": "closed",
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": 30,
                    "page": page,
                },
            )
            if not prs:
                break

            for pr in prs:
                if len(pairs) >= max_prs:
                    break

                # Must be actually merged
                if not pr.get("merged_at") or not pr.get("merge_commit_sha"):
                    continue

                title = pr.get("title", "")
                # Skip trivial/bot PRs
                if _SKIP_TITLE_PATTERNS.match(title):
                    logger.debug(f"    Skipping (trivial): {title}")
                    continue
                if pr.get("user", {}).get("type") == "Bot":
                    logger.debug(f"    Skipping (bot): {title}")
                    continue

                merge_sha = pr["merge_commit_sha"]

                # Check if this is a squash merge: merge commit has exactly 1 parent
                try:
                    commit_data = self._get(
                        f"{GITHUB_API}/repos/{repo_full_name}/commits/{merge_sha}"
                    )
                    parents = commit_data.get("parents", [])
                    if len(parents) != 1:
                        logger.debug(f"    Skipping (not squash, {len(parents)} parents): {title}")
                        continue

                    before_sha = parents[0]["sha"]
                except Exception as e:
                    logger.warning(f"    Failed to get commit {merge_sha}: {e}")
                    continue

                # Check diff stats
                stats = commit_data.get("stats", {})
                files_changed = len(commit_data.get("files", []))

                if files_changed < MIN_DIFF_FILES:
                    logger.debug(f"    Skipping (too few files: {files_changed}): {title}")
                    continue
                if files_changed > MAX_DIFF_FILES:
                    logger.debug(f"    Skipping (too many files: {files_changed}): {title}")
                    continue

                labels = [l.get("name", "") for l in pr.get("labels", [])]

                pair = CommitPair(
                    repo_full_name=repo_full_name,
                    pr_number=pr["number"],
                    pr_title=title,
                    pr_body=(pr.get("body") or "")[:2000],
                    pr_url=pr["html_url"],
                    pr_labels=labels,
                    before_sha=before_sha,
                    after_sha=merge_sha,
                    merge_commit_sha=merge_sha,
                    diff_files=[f["filename"] for f in commit_data.get("files", [])],
                    diff_stats={
                        "additions": stats.get("additions", 0),
                        "deletions": stats.get("deletions", 0),
                        "files_changed": files_changed,
                    },
                )

                logger.info(
                    f"    ✓ PR #{pr['number']}: {title} "
                    f"(+{stats.get('additions', 0)}/-{stats.get('deletions', 0)}, "
                    f"{files_changed} files)"
                )
                pairs.append(pair)
                time.sleep(0.3)

            page += 1
            time.sleep(0.5)

        return pairs

    def fetch_diff_text(self, repo_full_name: str, commit_sha: str) -> str:
        """Fetch the full diff text for a commit.

        Uses the .diff URL which returns plain text unified diff.
        """
        url = f"https://github.com/{repo_full_name}/commit/{commit_sha}.diff"
        resp = self.session.get(url)
        resp.raise_for_status()
        diff = resp.text

        # Truncate if too large
        if len(diff) > MAX_DIFF_CHARS:
            diff = diff[:MAX_DIFF_CHARS] + f"\n\n... [truncated, {len(resp.text)} chars total]"

        return diff

    def extract_commit_pairs(
        self,
        repo_full_name: str,
        max_pairs: int = 3,
    ) -> list[CommitPair]:
        """Full pipeline: find squash-merged PRs and fetch their diffs.

        Args:
            repo_full_name: e.g. "pocketbase/pocketbase"
            max_pairs: Maximum number of commit pairs to extract.

        Returns:
            List of CommitPair objects with all fields populated.
        """
        logger.info(f"Finding commit pairs for {repo_full_name}...")
        pairs = self.find_squash_merged_prs(repo_full_name, max_prs=max_pairs)

        for pair in pairs:
            logger.info(f"  Fetching diff for PR #{pair.pr_number}...")
            try:
                pair.diff_text = self.fetch_diff_text(
                    repo_full_name, pair.merge_commit_sha
                )
            except Exception as e:
                logger.warning(f"  Failed to fetch diff: {e}")
                pair.diff_text = ""

            time.sleep(0.3)

        return pairs


def save_commit_pairs(pairs: list[CommitPair], output_path: str) -> None:
    """Save commit pairs to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in pairs], f, indent=2)
    logger.info(f"Saved {len(pairs)} commit pairs to {output_path}")


def load_commit_pairs(input_path: str) -> list[CommitPair]:
    """Load commit pairs from JSON."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    return [CommitPair(**d) for d in data]
