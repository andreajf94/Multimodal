"""Reward functions for GRPO training of the RepoDesign model.

Components:
  1. format_compliance         (max 0.5)  — valid JSON with required fields
  2. format_partial            (max 0.25) — partial credit for near-valid structure
  3. existing_file_accuracy    (max 1.5)  — F1 of files_to_modify vs ground truth modified
  4. created_file_accuracy     (max 1.5)  — F1 of files_to_create vs ground truth created
  5. semantic_similarity       (max 3.0)  — sentence-embedding cosine similarity to teacher
  6. structural_quality        (max 0.5)  — plan structure (ticket/decision counts, etc.)

Format compliance gates all downstream rewards: if JSON parsing fails,
only format_partial contributes to the total.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Lazy-loaded sentence-transformer model (loaded once on first use)
_sentence_model = None

# Required top-level keys in a valid plan
_CORE_PLAN_KEYS = {"architecture_decisions", "tickets"}
_OPTIONAL_PLAN_KEYS_A = {"technology_choices"}       # synthetic pipeline
_OPTIONAL_PLAN_KEYS_B = {"implementation_summary"}   # commit-pair pipeline
REQUIRED_TICKET_KEYS = {"id", "title", "description"}
REQUIRED_DECISION_KEYS = {"dimension", "recommendation", "rationale"}


def _get_sentence_model():
    """Lazy-load the sentence-transformer model (once per process)."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded sentence-transformer model: all-MiniLM-L6-v2")
    return _sentence_model


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
            plan_keys = set(plan.keys())
            if not _CORE_PLAN_KEYS.issubset(plan_keys):
                scores.append(0.0)
                continue
            if not (_OPTIONAL_PLAN_KEYS_A & plan_keys or _OPTIONAL_PLAN_KEYS_B & plan_keys):
                scores.append(0.0)
                continue
            if not plan.get("architecture_decisions") or not plan.get("tickets"):
                scores.append(0.0)
                continue
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
    """Partial credit for near-valid JSON structure."""
    scores = []
    for text in completions:
        score = 0.0
        if "{" in text and "}" in text:
            score += 0.02
        if "[" in text and "]" in text:
            score += 0.01
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
            count = text.count(f'"{key}"')
            if count > 0:
                score += w * min(count, 5)
        brace_depth = min(text.count("{"), 20)
        score += brace_depth * 0.002
        n_quoted = len(re.findall(r'"[^"]{3,}"', text))
        score += min(n_quoted, 30) * 0.001
        scores.append(min(score, 0.25))
    return scores


# ---------------------------------------------------------------------------
# 3. Existing File Accuracy (F1 of files_to_modify vs ground truth)
# ---------------------------------------------------------------------------

def existing_file_accuracy(completions: list[str], diff_files: list[dict]) -> list[float]:
    """F1 score of files_to_modify vs ground truth modified files.

    Max score: 1.5.
    Precision = correct predictions / total predictions (rewards not hallucinating)
    Recall = correct predictions / total ground truth (rewards coverage)
    F1 = 2 * P * R / (P + R), scaled to [0, 1.5]
    """
    scores = []
    for text, diff_info in zip(completions, diff_files):
        gt_modified = {_normalize_path(f) for f in diff_info.get("modified", [])}

        plan = _parse_plan_json(text)
        if plan is not None:
            gen_modified = set()
            for ticket in plan.get("tickets", []):
                gen_modified.update(ticket.get("files_to_modify", []))
            gen_modified = {_normalize_path(f) for f in gen_modified if f}
        else:
            scores.append(0.0)
            continue

        if not gt_modified and not gen_modified:
            scores.append(1.5)
            continue

        if not gt_modified or not gen_modified:
            scores.append(0.0)
            continue

        tp = len(gt_modified & gen_modified)
        precision = tp / len(gen_modified) if gen_modified else 0.0
        recall = tp / len(gt_modified) if gt_modified else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1 * 1.5)
    return scores


# ---------------------------------------------------------------------------
# 4. Created File Accuracy (F1 of files_to_create vs ground truth)
# ---------------------------------------------------------------------------

def created_file_accuracy(completions: list[str], diff_files: list[dict]) -> list[float]:
    """F1 score of files_to_create vs ground truth created files.

    Max score: 1.5.
    When ground truth has no created files, this component is excluded
    (returns 0.0) to avoid rewarding minimal output.
    """
    scores = []
    for text, diff_info in zip(completions, diff_files):
        gt_created = {_normalize_path(f) for f in diff_info.get("created", [])}

        # If ground truth has no created files, exclude this component
        if not gt_created:
            scores.append(0.0)
            continue

        plan = _parse_plan_json(text)
        if plan is not None:
            gen_created = set()
            for ticket in plan.get("tickets", []):
                gen_created.update(ticket.get("files_to_create", []))
            gen_created = {_normalize_path(f) for f in gen_created if f}
        else:
            scores.append(0.0)
            continue

        if not gen_created:
            scores.append(0.0)
            continue

        tp = len(gt_created & gen_created)
        precision = tp / len(gen_created) if gen_created else 0.0
        recall = tp / len(gt_created) if gt_created else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1 * 1.5)
    return scores


# ---------------------------------------------------------------------------
# 5. Semantic Similarity (sentence embeddings)
# ---------------------------------------------------------------------------

def semantic_similarity(completions: list[str], teacher_plans: list[dict]) -> list[float]:
    """Cosine similarity of implementation_summary embeddings vs teacher.

    Max score: 3.0.
    Uses all-MiniLM-L6-v2 sentence-transformer for dense, meaningful similarity.
    """
    model = _get_sentence_model()

    scores = []
    for text, teacher in zip(completions, teacher_plans):
        teacher_summary = teacher.get("implementation_summary", "")
        if not teacher_summary or not isinstance(teacher_summary, str):
            scores.append(0.0)
            continue

        plan = _parse_plan_json(text)
        if plan is not None:
            gen_summary = plan.get("implementation_summary", "")
        else:
            scores.append(0.0)
            continue

        if not gen_summary:
            scores.append(0.0)
            continue

        try:
            embeddings = model.encode([teacher_summary, gen_summary])
            # Cosine similarity
            from numpy import dot
            from numpy.linalg import norm
            cos_sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
            # Clamp to [0, 1] (can be slightly negative for unrelated texts)
            cos_sim = max(0.0, float(cos_sim))
            scores.append(cos_sim * 3.0)
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            scores.append(0.0)

    return scores


# ---------------------------------------------------------------------------
# 6. Structural Quality
# ---------------------------------------------------------------------------

def structural_quality(completions: list[str]) -> list[float]:
    """Score the structural quality of the plan.

    Max score: 0.5.
    Rewards well-structured plans with appropriate counts of decisions and tickets.
    """
    scores = []
    for text in completions:
        plan = _parse_plan_json(text)
        if plan is None:
            scores.append(0.0)
            continue

        score = 0.0
        decisions = plan.get("architecture_decisions", [])
        tickets = plan.get("tickets", [])

        # Reward 3-8 architecture decisions (0.1 max)
        n_dec = len(decisions)
        if 3 <= n_dec <= 8:
            score += 0.1
        elif 1 <= n_dec <= 10:
            score += 0.05

        # Reward 4-10 tickets (0.1 max)
        n_tick = len(tickets)
        if 4 <= n_tick <= 10:
            score += 0.1
        elif 2 <= n_tick <= 15:
            score += 0.05

        # Reward ticket IDs following T-NNN pattern (0.1 max)
        if tickets:
            id_pattern = sum(1 for t in tickets if re.match(r"T-\d+", t.get("id", "")))
            score += 0.1 * (id_pattern / len(tickets))

        # Reward tickets having dependencies (0.1 max)
        if tickets:
            has_deps = sum(1 for t in tickets if t.get("dependencies"))
            score += 0.1 * min(has_deps / max(len(tickets) - 1, 1), 1.0)

        # Reward decisions having required fields (0.1 max)
        if decisions:
            complete = sum(
                1 for d in decisions
                if REQUIRED_DECISION_KEYS.issubset(d.keys())
            )
            score += 0.1 * (complete / len(decisions))

        scores.append(min(score, 0.5))
    return scores


# ---------------------------------------------------------------------------
# Combined reward (with format gating)
# ---------------------------------------------------------------------------

def compute_rewards(
    completions: list[str],
    file_manifests: list[list[str]],
    specs: list[dict],
    teacher_plans: list[dict],
    diff_files: list[dict],
    use_llm_judge: bool = True,  # Kept for backward compat, ignored
) -> list[dict]:
    """Compute all reward components for a batch of completions.

    Format compliance gates downstream rewards: if JSON parsing fails,
    only format_partial contributes to avoid noise injection.

    Max total: 7.25 (when created files exist in ground truth)
    Typical max: 5.75 (when no created files in ground truth)
    """
    fmt_exact = format_compliance(completions)
    fmt_partial = format_partial(completions)

    # Only compute content rewards for format-compliant completions
    # For non-compliant ones, these will all be 0
    existing_acc = existing_file_accuracy(completions, diff_files)
    created_acc = created_file_accuracy(completions, diff_files)
    sem_sim = semantic_similarity(completions, teacher_plans)
    struct = structural_quality(completions)

    # Check which completions parse as JSON at all (even if incomplete)
    json_parses = [_parse_plan_json(c) is not None for c in completions]

    results = []
    for i in range(len(completions)):
        if fmt_exact[i] > 0:
            # Fully format-compliant: full reward
            total = (fmt_exact[i] + fmt_partial[i] + existing_acc[i]
                     + created_acc[i] + sem_sim[i] + struct[i])
        elif json_parses[i]:
            # JSON parses but failed compliance (e.g. empty arrays):
            # still evaluate content so model prefers partial output over nothing
            total = (fmt_partial[i] + existing_acc[i]
                     + created_acc[i] + sem_sim[i] + struct[i])
        else:
            # Unparseable garbage: only partial credit
            total = fmt_partial[i]
            existing_acc[i] = 0.0
            created_acc[i] = 0.0
            sem_sim[i] = 0.0
            struct[i] = 0.0

        results.append({
            "format_compliance": fmt_exact[i],
            "format_partial": fmt_partial[i],
            "existing_file_accuracy": existing_acc[i],
            "created_file_accuracy": created_acc[i],
            "semantic_similarity": sem_sim[i],
            "structural_quality": struct[i],
            "total": total,
        })
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_plan_json(text: str) -> dict | None:
    """Try to parse JSON from model output (handles markdown blocks)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _extract_paths_regex(text: str) -> list[str]:
    """Extract file-path-like strings from raw text when JSON parsing fails."""
    pattern = r'["\']([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,10})["\']'
    matches = re.findall(pattern, text)
    paths = [m for m in matches if "/" in m or "\\" in m]
    unquoted = re.findall(r'(?<!\w)([a-zA-Z0-9_]+(?:/[a-zA-Z0-9_.]+){1,}\.(?:py|js|ts|yml|yaml|json|toml|md|txt|cfg|ini|sh|go|rs|java|rb|jsx|tsx))\b', text)
    paths.extend(unquoted)
    return list(set(paths))


def _extract_file_paths(plan: dict) -> tuple[list[str], list[str]]:
    """Extract file paths from a plan, split by must-exist vs new."""
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
    """Parse ground_truth_diff.txt to extract modified and created files."""
    modified = []
    created = []

    for line in diff_text.split('\n'):
        if line.startswith('diff --git'):
            match = re.search(r'a/(.*?) b/', line)
            if match:
                filepath = match.group(1)
                idx = diff_text.find(line)
                lines_after = diff_text[idx:idx+200].split('\n')[:5]
                if 'new file mode' in '\n'.join(lines_after):
                    created.append(filepath)
                else:
                    modified.append(filepath)

    return {"modified": modified, "created": created}
