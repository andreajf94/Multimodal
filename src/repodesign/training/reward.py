"""Reward functions for GRPO training of the RepoDesign model.

Adapted from IDEA-E Tunix GRPO reward patterns.

Components:
  1. format_compliance         (max 0.5)  — valid JSON with required fields
  2. format_partial            (max 0.25) — partial credit for near-valid structure
  3. existing_file_accuracy    (max 1.5)  — Jaccard similarity of files_to_modify vs ground_truth_diff modified files
  4. created_file_accuracy     (max 1.5)  — Jaccard similarity of files_to_create vs ground_truth_diff created files
  5. tfidf_similarity          (max 3.0)  — TF-IDF cosine similarity of implementation_summary to teacher
"""

from __future__ import annotations

import json
import logging
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)

# Global TF-IDF vectorizer (initialized lazily)
_tfidf_vectorizer = None

# Required top-level keys in a valid plan
# Core keys are always required; one of the optional sets must be present
_CORE_PLAN_KEYS = {"architecture_decisions", "tickets"}
_OPTIONAL_PLAN_KEYS_A = {"technology_choices"}       # synthetic pipeline
_OPTIONAL_PLAN_KEYS_B = {"implementation_summary"}   # commit-pair pipeline
REQUIRED_TICKET_KEYS = {"id", "title", "description"}
REQUIRED_DECISION_KEYS = {"dimension", "recommendation", "rationale"}


# ---------------------------------------------------------------------------
# 1. Format compliance (exact match)
# ---------------------------------------------------------------------------

def format_compliance(completions: list[str]) -> list[float]:
    """Score 0.5 if the completion is valid JSON with all required plan fields."""
    scores = []
    for text in completions:
        try:
            plan = _parse_plan_json(text)
            if plan is None:
                scores.append(0.0)
                continue
            # Check required keys
            plan_keys = set(plan.keys())
            if not _CORE_PLAN_KEYS.issubset(plan_keys):
                scores.append(0.0)
                continue
            # Accept either technology_choices or implementation_summary
            if not (_OPTIONAL_PLAN_KEYS_A & plan_keys or _OPTIONAL_PLAN_KEYS_B & plan_keys):
                scores.append(0.0)
                continue
            # Check that lists are non-empty
            if not plan.get("architecture_decisions") or not plan.get("tickets"):
                scores.append(0.0)
                continue
            # Check ticket structure
            first_ticket = plan["tickets"][0]
            if not REQUIRED_TICKET_KEYS.issubset(first_ticket.keys()):
                scores.append(0.0)
                continue
            scores.append(0.5)
        except Exception:
            scores.append(0.0)
    return scores


# ---------------------------------------------------------------------------
# 2. Format partial (approximate match)
# ---------------------------------------------------------------------------

def format_partial(completions: list[str]) -> list[float]:
    """Partial credit for near-valid JSON structure.

    More granular scoring to produce variance between completions:
      - Structural tokens: braces, brackets, colons
      - Key field mentions (weighted)
      - Depth of structure (nested objects)
      - Count of quoted strings (proxy for specificity)
    """
    scores = []
    for text in completions:
        score = 0.0
        # Credit for having JSON-like structure
        if "{" in text and "}" in text:
            score += 0.02
        if "[" in text and "]" in text:
            score += 0.01
        # Credit for key field names (fine-grained per key)
        key_weights = {
            "architecture_decisions": 0.02, "tickets": 0.02,
            "technology_choices": 0.02, "implementation_summary": 0.02,
            "files_to_modify": 0.015,
            "files_to_create": 0.015, "recommendation": 0.01,
            "rationale": 0.01, "dimension": 0.01, "description": 0.01,
            "estimated_effort": 0.01, "dependencies": 0.01,
            "alternatives_considered": 0.01,
        }
        for key, w in key_weights.items():
            # Count occurrences (more mentions = more structured)
            count = text.count(f'"{key}"')
            if count > 0:
                score += w * min(count, 5)  # cap at 5 mentions per key
        # Credit for nested object depth (more braces = deeper structure)
        brace_depth = min(text.count("{"), 20)
        score += brace_depth * 0.002
        # Credit for quoted strings (proxy for specificity)
        n_quoted = len(re.findall(r'"[^"]{3,}"', text))
        score += min(n_quoted, 30) * 0.001
        # Cap at 0.25
        scores.append(min(score, 0.25))
    return scores


# ---------------------------------------------------------------------------
# 3. Existing File Accuracy (files_to_modify vs ground truth modified files)
# ---------------------------------------------------------------------------

def existing_file_accuracy(completions: list[str], diff_files: list[dict]) -> list[float]:
    """Score based on Jaccard similarity of files_to_modify vs ground_truth_diff modified files.

    Max score: 1.5.
    Compares model's files_to_modify against actual modified files from the PR diff.
    Returns Jaccard similarity: |intersection| / |union| * 1.5
    """
    scores = []
    for text, diff_info in zip(completions, diff_files):
        # Extract ground truth modified files from diff
        gt_modified = {_normalize_path(f) for f in diff_info.get("modified", [])}

        # Extract generated files_to_modify from tickets
        plan = _parse_plan_json(text)
        if plan is not None:
            gen_modified = set()
            for ticket in plan.get("tickets", []):
                gen_modified.update(ticket.get("files_to_modify", []))
            gen_modified = {_normalize_path(f) for f in gen_modified if f}
        else:
            # JSON parsing failed - try regex fallback
            gen_modified = {_normalize_path(p) for p in _extract_paths_regex(text)}

        # If both are empty, full credit (correctly identified no modifications)
        if not gt_modified and not gen_modified:
            scores.append(1.5)
            continue
        
        # If ground truth has files but generated doesn't, score 0
        if gt_modified and not gen_modified:
            scores.append(0.0)
            continue

        # If generated has files but ground truth doesn't, score 0 (hallucinated)
        if not gt_modified and gen_modified:
            scores.append(0.0)
            continue

        # Jaccard similarity: intersection / union
        intersection = len(gt_modified & gen_modified)
        union = len(gt_modified | gen_modified)
        jaccard = intersection / union if union > 0 else 0.0
        scores.append(jaccard * 1.5)
    return scores


# ---------------------------------------------------------------------------
# 4. Created File Accuracy (files_to_create vs ground truth created files)
# ---------------------------------------------------------------------------

def created_file_accuracy(completions: list[str], diff_files: list[dict]) -> list[float]:
    """Score based on Jaccard similarity of files_to_create vs ground_truth_diff created files.

    Max score: 1.5.
    Compares model's files_to_create against actual created files from the PR diff.
    Returns Jaccard similarity: |intersection| / |union| * 1.5
    """
    scores = []
    for text, diff_info in zip(completions, diff_files):
        # Extract ground truth created files from diff
        gt_created = {_normalize_path(f) for f in diff_info.get("created", [])}

        # Extract generated files_to_create from tickets
        plan = _parse_plan_json(text)
        if plan is not None:
            gen_created = set()
            for ticket in plan.get("tickets", []):
                gen_created.update(ticket.get("files_to_create", []))
            gen_created = {_normalize_path(f) for f in gen_created if f}
        else:
            gen_created = set()

        # If both are empty, full credit (correctly identified no new files)
        if not gt_created and not gen_created:
            scores.append(1.5)
            continue
        
        # If ground truth has files but generated doesn't, score 0
        if gt_created and not gen_created:
            scores.append(0.0)
            continue

        # If generated has files but ground truth doesn't, score 0 (hallucinated new files)
        if not gt_created and gen_created:
            scores.append(0.0)
            continue

        # Jaccard similarity: intersection / union
        intersection = len(gt_created & gen_created)
        union = len(gt_created | gen_created)
        jaccard = intersection / union if union > 0 else 0.0
        scores.append(jaccard * 1.5)
    return scores


# ---------------------------------------------------------------------------
# 5. TF-IDF Similarity (implementation_summary text similarity to teacher)
# ---------------------------------------------------------------------------

def tfidf_similarity(completions: list[str], teacher_plans: list[dict]) -> list[float]:
    """Compute TF-IDF cosine similarity between generated and teacher implementation summaries.

    Max score: 3.0.
    Extracts implementation_summary from both generated and teacher plans.
    Returns cosine similarity * 3.0
    """
    global _tfidf_vectorizer
    
    # Initialize vectorizer if needed
    if _tfidf_vectorizer is None:
        _tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    
    scores = []
    for text, teacher in zip(completions, teacher_plans):
        # Extract teacher summary
        teacher_summary = teacher.get("implementation_summary", "")
        if not teacher_summary or not isinstance(teacher_summary, str):
            scores.append(0.0)
            continue
        
        # Extract generated summary
        plan = _parse_plan_json(text)
        if plan is not None:
            gen_summary = plan.get("implementation_summary", "")
        else:
            # JSON parsing failed - try to extract any text that looks like summary
            match = re.search(r'"implementation_summary"\s*:\s*"(.+?)"', text, re.DOTALL)
            gen_summary = match.group(1) if match else ""
        
        if not gen_summary:
            scores.append(0.0)
            continue
        
        # Compute TF-IDF cosine similarity
        try:
            # Fit on both documents then transform
            tfidf_matrix = _tfidf_vectorizer.fit_transform([teacher_summary, gen_summary])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            scores.append(float(similarity) * 3.0)
        except Exception as e:
            logger.warning(f"TF-IDF similarity failed: {e}")
            scores.append(0.0)
    
    return scores


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

def compute_rewards(
    completions: list[str],
    file_manifests: list[list[str]],
    specs: list[dict],
    teacher_plans: list[dict],
    diff_files: list[dict],
    use_llm_judge: bool = True,  # Kept for backward compat but ignored
) -> list[dict]:
    """Compute all reward components for a batch of completions.

    Args:
        completions: Model-generated plan texts
        file_manifests: Repo file lists (currently unused, kept for compat)
        specs: Feature specifications (currently unused, kept for compat)
        teacher_plans: Teacher reference plans
        diff_files: Ground truth diff files (modified/created) per example
        use_llm_judge: Deprecated, ignored

    Returns:
        List of dicts with per-component scores and total.
    """
    fmt_exact = format_compliance(completions)
    fmt_partial = format_partial(completions)
    existing_acc = existing_file_accuracy(completions, diff_files)
    created_acc = created_file_accuracy(completions, diff_files)
    tfidf_sim = tfidf_similarity(completions, teacher_plans)

    results = []
    for i in range(len(completions)):
        total = fmt_exact[i] + fmt_partial[i] + existing_acc[i] + created_acc[i] + tfidf_sim[i]
        results.append({
            "format_compliance": fmt_exact[i],
            "format_partial": fmt_partial[i],
            "existing_file_accuracy": existing_acc[i],
            "created_file_accuracy": created_acc[i],
            "tfidf_similarity": tfidf_sim[i],
            "total": total,
        })
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_plan_json(text: str) -> dict | None:
    """Try to parse JSON from model output (handles markdown blocks)."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _extract_paths_regex(text: str) -> list[str]:
    """Extract file-path-like strings from raw text when JSON parsing fails.

    Matches patterns like: src/foo/bar.py, ./config/settings.yml, etc.
    """
    # Match quoted strings that look like file paths (contain / or \ and an extension)
    pattern = r'["\']([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})["\']'
    matches = re.findall(pattern, text)
    # Filter to things that look like real file paths (have at least one directory separator)
    paths = [m for m in matches if "/" in m or "\\" in m]
    # Also match unquoted paths with directory structure
    unquoted = re.findall(r'(?<!\w)([a-zA-Z0-9_]+(?:/[a-zA-Z0-9_.]+){1,}\.(?:py|js|ts|yml|yaml|json|toml|md|txt|cfg|ini|sh|go|rs|java|rb|jsx|tsx))\b', text)
    paths.extend(unquoted)
    return list(set(paths))


def _extract_file_paths(plan: dict) -> tuple[list[str], list[str]]:
    """Extract file paths from a plan, split by must-exist vs new.

    Returns:
        (must_exist, new_files) where must_exist are files_to_modify
        (should be in manifest) and new_files are files_to_create
        (should NOT be in manifest).

    Note: architecture_decisions.files_affected is informational and may
    reference both existing and new files, so it is NOT grounding-checked.
    Only tickets.files_to_modify is checked against the manifest.
    """
    must_exist: set[str] = set()
    new_files: set[str] = set()
    for ticket in plan.get("tickets", []):
        must_exist.update(ticket.get("files_to_modify", []))
        new_files.update(ticket.get("files_to_create", []))
    return list(must_exist), list(new_files)


def _normalize_path(path: str) -> str:
    """Normalize a file path for comparison."""
    p = path.strip().replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    if p.startswith("/"):
        p = p[1:]
    return p


def parse_diff_files(diff_text: str) -> dict[str, list[str]]:
    """Parse ground_truth_diff.txt to extract modified and created files.
    
    Returns:
        Dict with "modified" and "created" lists of file paths.
    """
    modified = []
    created = []
    
    # Parse git diff headers: "diff --git a/path b/path"
    for line in diff_text.split('\n'):
        if line.startswith('diff --git'):
            # Extract path: "diff --git a/path b/path"
            match = re.search(r'a/(.*?) b/', line)
            if match:
                filepath = match.group(1)
                
                # Check if it's a new file by looking ahead
                idx = diff_text.find(line)
                lines_after = diff_text[idx:idx+200].split('\n')[:5]
                if 'new file mode' in '\n'.join(lines_after):
                    created.append(filepath)
                else:
                    modified.append(filepath)
    
    return {"modified": modified, "created": created}
