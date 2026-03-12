"""Training data generation from commit pairs: teacher explains real code changes.

Instead of inventing synthetic features + plans, this module:
  1. Takes a CommitPair (before/after SHA, diff, PR description)
  2. Takes the RepoIR extracted at the "before" commit
  3. Asks the teacher model to explain HOW and WHY the changes were made
  4. Produces a structured teacher explanation grounded in real file paths

The training example becomes:
  - Prompt: "Here's repo at commit X. We want to add feature Y. Describe how."
  - Ground truth: The real diff + teacher explanation
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeepSeek client (reused from data_gen)
# ---------------------------------------------------------------------------

def _call_deepseek(system_prompt: str, user_prompt: str, model: str = "deepseek-chat", max_tokens: int = 4096) -> str:
    """Call DeepSeek and return the text response."""
    import openai

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response (handles markdown code blocks)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not extract JSON from response:\n{text[:500]}")


# ---------------------------------------------------------------------------
# Feature spec extraction from PR metadata
# ---------------------------------------------------------------------------

SPEC_FROM_PR_SYSTEM = """You are a product manager. Given a Pull Request title, description, and list of changed files, write a concise feature specification that a developer would receive BEFORE implementing this change.

Return a JSON object:
{
  "feature_name": "short name for the feature",
  "description": "2-3 sentence description of what to build, written as a forward-looking request (not past tense)",
  "functional_requirements": ["3-6 specific, actionable requirements"]
}

Guidelines:
- Write as if the feature hasn't been built yet — this is a request, not a changelog
- Be specific about WHAT to build but don't reveal HOW (the model needs to figure that out)
- Reference the general area of the codebase but don't list exact file paths
- Return ONLY the JSON object"""


def generate_spec_from_pr(pr_title: str, pr_body: str, diff_files: list[str]) -> dict:
    """Generate a forward-looking feature spec from PR metadata.

    This transforms a PR (past tense: "Added X") into a spec (future tense: "Add X").
    """
    user_prompt = f"""## Pull Request
Title: {pr_title}

Description:
{pr_body[:1500] if pr_body else '(no description)'}

## Files Changed ({len(diff_files)} files)
{chr(10).join(diff_files[:30])}
"""
    raw = _call_deepseek(SPEC_FROM_PR_SYSTEM, user_prompt, model="deepseek-chat")
    return _extract_json(raw)


# ---------------------------------------------------------------------------
# Teacher explanation from real diff
# ---------------------------------------------------------------------------

TEACHER_EXPLAIN_SYSTEM = """You are a senior software architect explaining real code changes to a junior developer. Given:
1. A summary of the codebase BEFORE the change
2. The feature that was requested
3. The actual unified diff showing what was changed

Explain the implementation decisions in a structured JSON format:

{
  "architecture_decisions": [
    {
      "dimension": "string (e.g., api_design, data_modeling, error_handling, testing)",
      "recommendation": "what was done and why",
      "rationale": "technical reasoning for this approach",
      "alternatives_considered": ["what else could have been done"],
      "files_affected": ["actual file paths from the diff"]
    }
  ],
  "tickets": [
    {
      "id": "T-001",
      "title": "logical unit of work",
      "description": "what this piece accomplishes",
      "files_to_modify": ["files that were modified"],
      "files_to_create": ["files that were created"],
      "estimated_effort": "small | medium | large"
    }
  ],
  "implementation_summary": "2-3 paragraph explanation of the overall approach, key patterns used, and how the changes integrate with the existing codebase"
}

Guidelines:
- ALL file paths must come from the actual diff — these are real changes
- Group related changes into logical tickets
- Explain WHY each decision was made, not just WHAT was done
- Note any interesting patterns, trade-offs, or design choices
- Return ONLY the JSON object"""


def _categorize_diff_files(diff_text: str, file_manifest: set[str]) -> tuple[list[str], list[str]]:
    """Categorize changed files into modify vs create based on diff headers and manifest.
    
    Args:
        diff_text: The unified diff.
        file_manifest: Set of existing file paths from RepoIR.
    
    Returns:
        (files_to_modify, files_to_create)
    """
    files_to_modify = []
    files_to_create = []
    
    # Parse diff to find new files (look for "new file mode" in diff headers)
    current_file = None
    is_new_file = False
    
    for line in diff_text.split('\n'):
        if line.startswith('diff --git '):
            # Extract filename: diff --git a/path/file.txt b/path/file.txt
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[2][2:]  # Remove 'a/' prefix
                is_new_file = False
        elif line.startswith('new file mode') and current_file:
            is_new_file = True
        elif line.startswith('---') and current_file:
            # End of header, categorize this file
            if is_new_file or current_file not in file_manifest:
                files_to_create.append(current_file)
            else:
                files_to_modify.append(current_file)
            current_file = None
            is_new_file = False
    
    return files_to_modify, files_to_create


def generate_teacher_explanation(
    repo_ir_summary: str,
    spec: dict,
    diff_text: str,
    diff_files: list[str],
    file_manifest: set[str],
) -> dict:
    """Generate a teacher explanation of real code changes.

    Args:
        repo_ir_summary: Condensed RepoIR summary (before state).
        spec: Feature spec derived from PR metadata.
        diff_text: The actual unified diff.
        diff_files: List of changed file paths.
        file_manifest: Set of existing file paths for categorization.

    Returns:
        Structured explanation dict.
    """
    # Categorize files based on diff headers and manifest
    files_to_modify, files_to_create = _categorize_diff_files(diff_text, file_manifest)
    
    # Truncate diff if needed to fit context
    diff_for_prompt = diff_text
    if len(diff_for_prompt) > 40_000:
        diff_for_prompt = diff_for_prompt[:40_000] + "\n\n... [diff truncated]"

    user_prompt = f"""## Codebase (before the change)
{repo_ir_summary}

## Feature Request
{json.dumps(spec, indent=2)}

## Files to MODIFY (already exist in repo)
{chr(10).join(files_to_modify) if files_to_modify else '(none)'}

## Files to CREATE (new files)
{chr(10).join(files_to_create) if files_to_create else '(none)'}

## Actual Diff
```diff
{diff_for_prompt}
```

Explain the implementation decisions made in this change. 

CRITICAL REQUIREMENTS:
1. Use ONLY files from the lists above - files_to_modify must only contain existing files, files_to_create must only contain new files
2. Copy file paths EXACTLY as shown above, character-for-character, including any hashes, version numbers, or generated names
3. Do NOT invent, modify, or "clean up" file paths - use the exact paths from the diff"""

    raw = _call_deepseek(
        TEACHER_EXPLAIN_SYSTEM,
        user_prompt,
        model="deepseek-chat",
        max_tokens=8192,
    )
    teacher_plan = _extract_json(raw)
    
    # Extract actual diff files for validation
    diff_file_set = set(diff_files)
    
    # Helper to fix hallucinated hashes: ui/dist/assets/File.WRONG.js -> ui/dist/assets/File.CORRECT.js
    def fix_hallucinated_filename(filename: str) -> str:
        """If filename has wrong hash, replace with correct one from diff."""
        if filename in diff_file_set or filename in file_manifest:
            return filename
        
        # Check if it's a hashed filename that got the hash wrong
        # Pattern: path/basename.HASH.ext where HASH is 8 hex chars
        import re
        parts = filename.rsplit('.', 2)
        if len(parts) == 3 and re.match(r'^[a-f0-9]{8}$', parts[1]):
            # Has hash format, try to find correct version in diff
            base_pattern = parts[0]  # e.g., "ui/dist/assets/AuthMethodsDocs"
            ext = parts[2]  # e.g., "js"
            
            for diff_file in diff_file_set:
                if diff_file.startswith(base_pattern + '.') and diff_file.endswith('.' + ext):
                    return diff_file
        
        return filename
    
    # Post-process to fix any categorization errors and hallucinated filenames
    for ticket in teacher_plan.get('tickets', []):
        modify = ticket.get('files_to_modify', [])
        create = ticket.get('files_to_create', [])
        
        # Fix hallucinated hashes
        modify = [fix_hallucinated_filename(f) for f in modify]
        create = [fix_hallucinated_filename(f) for f in create]
        
        # Fix categorization
        fixed_modify = [f for f in modify if f in file_manifest]
        fixed_create = [f for f in create if f not in file_manifest]
        
        # Move incorrectly categorized files
        wrongly_in_modify = [f for f in modify if f not in file_manifest]
        wrongly_in_create = [f for f in create if f in file_manifest]
        
        ticket['files_to_modify'] = fixed_modify + wrongly_in_create
        ticket['files_to_create'] = fixed_create + wrongly_in_modify
    
    return teacher_plan


# ---------------------------------------------------------------------------
# Full pipeline: CommitPair + RepoIR → training example
# ---------------------------------------------------------------------------

def generate_commit_pair_example(
    commit_pair_data: dict,
    repo_ir: dict,
    output_dir: str,
) -> dict | None:
    """Generate a complete training example from a commit pair.

    Args:
        commit_pair_data: Dict from CommitPair (has pr_title, pr_body, diff_text, etc.)
        repo_ir: RepoIR dict extracted at the before_sha commit.
        output_dir: Directory to save spec.json and teacher_plan.json.

    Returns:
        Dict with training example data, or None on failure.
    """
    from repodesign.training.data_gen import summarize_repo_ir_for_prompt

    repo_name = repo_ir.get("repo_metadata", {}).get("name", "unknown")
    file_manifest = repo_ir.get("file_manifest", [])
    summary = summarize_repo_ir_for_prompt(repo_ir)

    pr_title = commit_pair_data.get("pr_title", "")
    pr_body = commit_pair_data.get("pr_body", "")
    diff_text = commit_pair_data.get("diff_text", "")
    diff_files = commit_pair_data.get("diff_files", [])

    # Step 1: Generate a forward-looking spec from the PR metadata
    logger.info(f"  Generating spec from PR #{commit_pair_data.get('pr_number')}...")
    try:
        spec = generate_spec_from_pr(pr_title, pr_body, diff_files)
        spec["project_name"] = repo_name
        spec["source_pr"] = commit_pair_data.get("pr_url", "")
    except Exception as e:
        logger.error(f"  Spec generation failed: {e}")
        return None

    # Step 2: Generate teacher explanation from the real diff
    logger.info(f"  Generating teacher explanation...")
    try:
        teacher_plan = generate_teacher_explanation(
            summary, spec, diff_text, diff_files, set(file_manifest)
        )
        # Add metadata
        teacher_plan["source_pr"] = commit_pair_data.get("pr_url", "")
        teacher_plan["before_sha"] = commit_pair_data.get("before_sha", "")
        teacher_plan["after_sha"] = commit_pair_data.get("after_sha", "")
        teacher_plan["diff_stats"] = commit_pair_data.get("diff_stats", {})
    except Exception as e:
        logger.error(f"  Teacher explanation failed: {e}")
        return None

    # Save outputs
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, "spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    with open(os.path.join(output_dir, "teacher_plan.json"), "w", encoding="utf-8") as f:
        json.dump(teacher_plan, f, indent=2)

    # Save diff as reference
    with open(os.path.join(output_dir, "ground_truth_diff.txt"), "w", encoding="utf-8") as f:
        f.write(diff_text)

    logger.info(f"  Saved spec + teacher plan + diff to {output_dir}")

    return {
        "repo_name": repo_name,
        "repo_ir_summary": summary,
        "spec": spec,
        "teacher_plan": teacher_plan,
        "file_manifest": file_manifest,
        "diagram_paths": repo_ir.get("diagram_paths", []),
        "commit_pair": {
            "before_sha": commit_pair_data.get("before_sha"),
            "after_sha": commit_pair_data.get("after_sha"),
            "pr_number": commit_pair_data.get("pr_number"),
            "pr_url": commit_pair_data.get("pr_url"),
            "diff_files": diff_files,
        },
    }
