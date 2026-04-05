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
# 4. Created File Accuracy (fuzzy nearest-neighbor matching)
# ---------------------------------------------------------------------------

def _path_similarity(path_a: str, path_b: str) -> float:
    """Score similarity between two file paths using component-level matching.

    Returns a value in [0, 1]:
      - 1.0 = exact match
      - 0.7+ = same directory, similar filename
      - 0.3+ = same parent dir, different filename
      - 0.1+ = same extension, different location
      - 0.0  = completely different

    Scoring:
      - Directory overlap (60% weight): fraction of shared path components
      - Filename similarity (25% weight): character-level similarity
      - Extension match (15% weight): same file type
    """
    from difflib import SequenceMatcher

    parts_a = path_a.split("/")
    parts_b = path_b.split("/")
    dir_a, file_a = parts_a[:-1], parts_a[-1]
    dir_b, file_b = parts_b[:-1], parts_b[-1]

    # Directory overlap: shared components / max components
    if dir_a or dir_b:
        shared_dirs = sum(1 for a, b in zip(dir_a, dir_b) if a == b)
        max_dirs = max(len(dir_a), len(dir_b), 1)
        dir_score = shared_dirs / max_dirs
    else:
        dir_score = 1.0  # both in root

    # Filename similarity (SequenceMatcher ratio)
    name_score = SequenceMatcher(None, file_a, file_b).ratio()

    # Extension match
    ext_a = file_a.rsplit(".", 1)[-1] if "." in file_a else ""
    ext_b = file_b.rsplit(".", 1)[-1] if "." in file_b else ""
    ext_score = 1.0 if ext_a == ext_b else 0.0

    return 0.60 * dir_score + 0.25 * name_score + 0.15 * ext_score


def _best_alignment_score(gt_paths: set[str], gen_paths: set[str]) -> float:
    """Compute F1 using nearest-neighbor alignment with fuzzy path matching.

    For each GT file, find the best-matching generated file (and vice versa).
    A match scores between 0 and 1 based on path similarity.
    Exact matches still score 1.0; partial matches get proportional credit.
    """
    if not gt_paths or not gen_paths:
        return 0.0

    gt_list = sorted(gt_paths)
    gen_list = sorted(gen_paths)

    # For recall: for each GT file, best similarity to any generated file
    recall_scores = []
    for gt in gt_list:
        best = max(_path_similarity(gt, g) for g in gen_list)
        recall_scores.append(best)

    # For precision: for each generated file, best similarity to any GT file
    precision_scores = []
    for g in gen_list:
        best = max(_path_similarity(gt, g) for gt in gt_list)
        precision_scores.append(best)

    precision = sum(precision_scores) / len(precision_scores)
    recall = sum(recall_scores) / len(recall_scores)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def created_file_accuracy(completions: list[str], diff_files: list[dict]) -> list[float]:
    """Fuzzy F1 of files_to_create vs ground truth created files.

    Max score: 1.5.
    Uses nearest-neighbor alignment with component-level path similarity
    so that predicting the right directory + similar filename gets partial
    credit instead of zero.

    When ground truth has no created files, returns 0.0.
    """
    scores = []
    for text, diff_info in zip(completions, diff_files):
        gt_created = {_normalize_path(f) for f in diff_info.get("created", [])}

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

        f1 = _best_alignment_score(gt_created, gen_created)
        scores.append(f1 * 1.5)
    return scores


# ---------------------------------------------------------------------------
# 5. Semantic Similarity (sentence embeddings)
# ---------------------------------------------------------------------------

def _cosine_sim(model, text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using sentence-transformers."""
    from numpy import dot
    from numpy.linalg import norm
    embeddings = model.encode([text_a, text_b])
    sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
    return max(0.0, float(sim))


def semantic_similarity(completions: list[str], teacher_plans: list[dict]) -> list[float]:
    """Component-level semantic similarity against teacher plan.

    Max score: 3.0.
    Three sub-components (weighted):
      - Summary similarity (50%):  implementation_summary vs teacher summary
      - Decision similarity (30%): best-match rationale across architecture_decisions
      - Ticket similarity (20%):   best-match description across tickets

    This gives more targeted signal than comparing summaries alone.
    """
    model = _get_sentence_model()

    scores = []
    for text, teacher in zip(completions, teacher_plans):
        plan = _parse_plan_json(text)
        if plan is None:
            scores.append(0.0)
            continue

        total = 0.0

        # --- Summary similarity (50% of 3.0 = 1.5 max) ---
        teacher_summary = teacher.get("implementation_summary", "")
        gen_summary = plan.get("implementation_summary", "")
        if teacher_summary and gen_summary:
            try:
                total += _cosine_sim(model, teacher_summary, gen_summary) * 1.5
            except Exception:
                pass

        # --- Decision similarity (30% of 3.0 = 0.9 max) ---
        teacher_decisions = teacher.get("architecture_decisions", [])
        gen_decisions = plan.get("architecture_decisions", [])
        if teacher_decisions and gen_decisions:
            try:
                teacher_rationales = [d.get("rationale", "") or d.get("recommendation", "")
                                      for d in teacher_decisions if isinstance(d, dict)]
                gen_rationales = [d.get("rationale", "") or d.get("recommendation", "")
                                  for d in gen_decisions if isinstance(d, dict)]
                teacher_rationales = [r for r in teacher_rationales if r]
                gen_rationales = [r for r in gen_rationales if r]

                if teacher_rationales and gen_rationales:
                    # Best-match: for each teacher decision, find best generated match
                    all_texts = teacher_rationales + gen_rationales
                    embs = model.encode(all_texts)
                    t_embs = embs[:len(teacher_rationales)]
                    g_embs = embs[len(teacher_rationales):]

                    from numpy import dot
                    from numpy.linalg import norm
                    match_scores = []
                    for t_emb in t_embs:
                        best = max(
                            float(dot(t_emb, g_emb) / (norm(t_emb) * norm(g_emb)))
                            for g_emb in g_embs
                        )
                        match_scores.append(max(0.0, best))
                    total += (sum(match_scores) / len(match_scores)) * 0.9
            except Exception:
                pass

        # --- Ticket similarity (20% of 3.0 = 0.6 max) ---
        teacher_tickets = teacher.get("tickets", [])
        gen_tickets = plan.get("tickets", [])
        if teacher_tickets and gen_tickets:
            try:
                teacher_descs = [t.get("description", "") or t.get("title", "")
                                 for t in teacher_tickets if isinstance(t, dict)]
                gen_descs = [t.get("description", "") or t.get("title", "")
                             for t in gen_tickets if isinstance(t, dict)]
                teacher_descs = [d for d in teacher_descs if d]
                gen_descs = [d for d in gen_descs if d]

                if teacher_descs and gen_descs:
                    all_texts = teacher_descs + gen_descs
                    embs = model.encode(all_texts)
                    t_embs = embs[:len(teacher_descs)]
                    g_embs = embs[len(teacher_descs):]

                    from numpy import dot
                    from numpy.linalg import norm
                    match_scores = []
                    for t_emb in t_embs:
                        best = max(
                            float(dot(t_emb, g_emb) / (norm(t_emb) * norm(g_emb)))
                            for g_emb in g_embs
                        )
                        match_scores.append(max(0.0, best))
                    total += (sum(match_scores) / len(match_scores)) * 0.6
            except Exception:
                pass

        scores.append(total)
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
