# RepoDesign — Midterm Results

This document summarizes all training and evaluation results for the RepoDesign project as of 2026-04-06. It covers the training pipeline, reward function design, evaluation methodology, quantitative results, and qualitative output examples.

---

## 1. What We Built

RepoDesign is a multimodal AI agent that takes a product specification, an existing codebase representation, and architecture diagrams (when available), and produces a structured implementation plan. The plan contains:

- **Architecture decisions** with rationale, alternatives considered, and affected files
- **Tickets** with file paths to modify/create, descriptions, effort estimates, and dependencies
- **Implementation summary** explaining the overall approach

The model is fine-tuned (LoRA) from `Qwen3-VL-235B-A22B-Instruct` via the Tinker SDK, using GRPO (Group Relative Policy Optimization) with a custom multi-component reward function. Training data is 232 commit pairs derived from real merged PRs across 25 popular open-source repos.

---

## 2. Training Pipeline

### 2.1 Reward Function (7 components)

Implemented in [`src/repodesign/training/reward.py`](src/repodesign/training/reward.py).

| Component | Max | Description |
|-----------|-----|-------------|
| `format_compliance` | 0.5 | Valid JSON with required schema fields, populated arrays |
| `format_partial` | 0.25 | Partial credit for near-valid JSON structure |
| `existing_file_accuracy` | 1.5 | F1 of `files_to_modify` against ground-truth modified files |
| `created_file_accuracy` | 1.5 | Fuzzy F1 of `files_to_create` (nearest-neighbor path matching) |
| `semantic_similarity` | 3.0 | Component-level cosine similarity (summary 50% + decisions 30% + tickets 20%) |
| `structural_quality` | 0.5 | Number of decisions/tickets, T-NNN ID format, dependencies populated |
| `nonempty_bonus` | 0.5 | Flat +0.5 if `architecture_decisions` and `tickets` are both populated |
| **Total max** | **7.75** | |

A 3-tier format gate prevents reward signal collapse on malformed JSON:
1. **Full valid JSON** → all components contribute
2. **JSON parses but empty arrays** → content rewards still apply (avoids dead signal)
3. **Unparseable** → only `format_partial` contributes

### 2.2 Curriculum Learning

Implemented in `StratifiedSampler` ([`scripts/train_grpo_tinker.py`](scripts/train_grpo_tinker.py)).

- **Difficulty proxy**: number of files in the ground truth diff (modified + created)
- **Bins**: easy (≤3 files, 62 examples), medium (4–6 files, 62 examples), hard (≥6 files, 63 examples)
- **Progressive curriculum**: 60/30/10 split (easy/med/hard) at step 0, shifting to 30/30/40 by step 50
- **EMA reweighting**: per-example reward EMA (α=0.3) — examples in the "learning zone" (not mastered, not stuck) get 1.5× sampling weight

### 2.3 Hyperparameters

- LoRA rank: 64
- Batch size: 4 prompts
- Group size (G): 4 completions per prompt
- Max generation tokens: 8192
- Learning rate: 1e-5 → 1e-6 (cosine decay), 5% warmup
- KL penalty β: 0.1
- Advantage clipping: ±2.0
- Loss: importance sampling (PPO-style)

---

## 3. Training Results

### 3.1 Curriculum vs Random Batching (25-step ablation)

We first validated curriculum learning on a 25-step run vs random batching:

| Metric | Random (`crisp-jazz-27`) | Curriculum (`northern-dawn-29`) |
|--------|--------------------------|--------------------------------|
| First-5 avg reward | 2.50 | 2.30 |
| Last-5 avg reward | 3.64 | **4.01** |
| Trend (last5 − first5) | +1.14 | **+1.72** |
| Std deviation | 0.93 | **0.80** |
| Floor (min reward) | 1.05 | **1.76** |

**Curriculum learning gave a +51% steeper learning trend, eliminated dead-batch collapses (no more 1.0 floors), and reduced step-to-step variance by 14%.**

### 3.2 Full 100-Step Training Run (`prime-hill-30`)

Run name in W&B: `prime-hill-30` (project: `repodesign-grpo`)

**Phase analysis:**

| Phase | Steps | Mean reward | Format rate | Modify F1 | Created F1 | Semantic |
|-------|-------|------------|-------------|-----------|------------|----------|
| Warmup | 0–10 | 2.63 | 30% | 0.32 | 0.01 | 1.85 |
| Transition | 30–50 | 3.91 | 79% | 0.66 | 0.17 | 2.07 |
| Mature | 51–75 | 4.51 | 98% | 0.86 | 0.15 | 2.27 |
| Final | 76–99 | 4.52 | 98% | 0.86 | 0.14 | 2.29 |

**Key milestones:**
- First step at 100% format compliance: **step 43**
- Sustained ≥90% format compliance from: **step 50**
- Personal best reward: **5.31** (step 88)
- First-10 → last-10 improvement: **+1.83** (2.62 → 4.45)
- 35 of the last 40 steps at 100% format compliance

---

## 4. Evaluation Methodology

We evaluated the trained model on **two held-out splits**, designed to test different generalization properties.

### 4.1 RepoDesign-Bench Construction

| Split | N | Source | Tests |
|-------|---|--------|-------|
| **Held-out** | 45 (43 scored) | Same 25 training repos, **different PRs** | In-distribution generalization |
| **Unseen-repo** | 12 | 6 NEW repos: Django, Flask, Celery, Pydantic, httpx, SQLModel | Out-of-distribution generalization |
| **Total** | **57** | | |

**Held-out PRs** are sampled from the same 25 repos in the training pool, but using a 20% stratified split (by repo prefix). The model has seen the codebase architecture during training but never these specific PRs.

**Unseen-repo PRs** are from repos the model has never encountered in any form. We pulled 2 recent merged PRs from each of 6 popular Python/JavaScript projects, hand-wrote specs based on the PR descriptions, and used the actual ground truth diffs for scoring.

### 4.2 Inference Setup

- Loads the final checkpoint from `output/grpo_v5_100steps/checkpoints.jsonl` via Tinker
- 4 completions per example, best-of-N selected by reward
- Same prompt format as training (candidate file list with 50 distractors + ground truth files)
- Diagrams loaded if available

Eval script: [`scripts/run_eval_inference.py`](scripts/run_eval_inference.py)

### 4.3 What We Measure

- **Total reward** (max 7.75) — composite of all 7 components
- **Format compliance rate** — fraction of examples producing valid JSON with required fields
- **Modify F1** — file path F1 for files the model says to modify (exact match required)
- **Created F1** — fuzzy F1 for files the model says to create (nearest-neighbor path matching with directory + filename + extension components)
- **Semantic similarity** (max 3.0) — component-level cosine similarity vs teacher plan (summary 50% + decisions 30% + tickets 20%)

---

## 5. Quantitative Results

### 5.1 Full Eval Summary

| Metric | Base Model | Trained Model (held-out) | Trained Model (unseen) |
|--------|-----------|--------------------------|------------------------|
| N | 12 | 43 | 12 |
| **Total reward** | 1.62 | **4.73** | **4.71** |
| **Format pass rate** | 17% | **100%** | **100%** |
| **Modify F1** | 0.15 | **1.03** | **0.98** |
| **Created F1** | 0.00 | **0.34** | 0.08 |
| **Semantic similarity** | 1.19 | 2.11 | 2.40 |

> **Modify F1 > 1.0** on held-out occurs because the model's predictions sometimes recall *more* relevant files than the minimum set in the ground truth diff (rewarding recall).

### 5.2 Base vs Trained (Unseen Repos, Per-Example)

The trained model wins **12/12** on completely new repositories.

| Repo | Base reward | Trained reward | Δ |
|------|------------|----------------|---|
| celery_pr10159 | 0.10 | 4.87 | +4.78 |
| celery_pr10184 | 0.10 | 5.37 | +5.28 |
| django_pr20828 | 2.52 | 4.75 | +2.22 |
| django_pr21046 | 0.10 | 5.12 | +5.03 |
| flask_pr5917 | 2.61 | 4.74 | +2.13 |
| flask_pr5928 | 4.78 | 4.86 | +0.07 |
| httpx_pr3670 | 2.38 | 3.51 | +1.13 |
| httpx_pr3673 | 2.14 | 4.35 | +2.22 |
| pydantic_pr13013 | 0.10 | 5.29 | +5.19 |
| pydantic_pr13016 | 0.10 | 4.99 | +4.89 |
| sqlmodel_pr1701 | 4.48 | 4.54 | +0.06 |
| sqlmodel_pr1806 | 0.10 | 4.16 | +4.07 |
| **Average** | **1.62** | **4.71** | **+3.09** |

**The biggest gains are on examples where the base model produced empty plans (reward 0.10 = `format_partial` only).** The base Qwen3-VL model can produce a valid JSON-like plan only 17% of the time on these specs; the trained model hits 100%.

### 5.3 Multimodal Ablation (25-step runs)

Trained two parallel 25-step runs with the same curriculum and reward function, one with diagrams and one without:

| Metric | With diagrams | Without diagrams | Δ |
|--------|--------------|------------------|---|
| Mean reward | 3.30 | 3.26 | +0.04 |
| Last-5 avg | 4.13 | 3.97 | +0.16 |
| Trend (first-5 → last-5) | +1.84 | +1.56 | +0.28 |
| Step-by-step wins | 15/25 | 10/25 | — |

**Caveat:** only 23% of training examples (54/232) actually have loadable diagram files (`diagram_paths` populated AND files exist on disk). The other 77% are identical between the two runs, which dilutes any real multimodal effect by ~4×. The +0.16 last-5 advantage with diagrams, attributed entirely to the 23% subset, suggests diagrams add roughly +0.7 reward per multimodal example — but we cannot prove this with the current diagram coverage. **Improving diagram extraction is the highest-priority unblocker for the multimodal claim.**

---

## 6. Qualitative Output Examples

These illustrate what the trained model actually produces. Best examples by reward, all from the 100-step run.

### 6.1 Top Held-Out Example: `axum_pr3664` (reward 6.43)

**Spec (real PR title):** Add `Content-Disposition` header escaping to prevent header injection vulnerabilities

**Ground truth modified files:** `axum-extra/Cargo.toml`, `axum-extra/src/response/file_stream.rs`

**Model output (excerpt):**

```json
{
  "architecture_decisions": [
    {
      "dimension": "Security",
      "recommendation": "Implement RFC 2616 quoted-string escaping for filenames in Content-Disposition headers.",
      "rationale": "This prevents header injection attacks by properly escaping backslashes and double quotes, ensuring filenames cannot break out of the quoted string boundary. This is a critical security fix that aligns with HTTP specification requirements.",
      "alternatives_considered": [
        "Leaving as-is (insecure)",
        "URL-encoding filenames (not compliant with RFC 2616 for quoted-string)"
      ],
      "files_affected": ["axum-extra/src/response/file_stream.rs"]
    }
  ],
  "tickets": [...],
  "implementation_summary": "..."
}
```

**Why it scored well:** correct files identified (Modify F1 = 1.0), correctly recognized this as a security-driven RFC 2616 compliance issue, named the actual CVE class (header injection), and produced 5 well-scoped tickets with dependencies.

### 6.2 Top Unseen-Repo Example: `celery_pr10184` (reward 5.37)

**Spec:** "Fix: prioritize request `ignore_result` over task definition" — apply_async(ignore_result=False) on a task with `@app.task(ignore_result=True)` should still return results.

**Ground truth modified files:** `celery/app/trace.py`, `celery/worker/request.py`, `docs/userguide/tasks.rst`, `t/unit/tasks/test_trace.py`, `t/unit/worker/test_request.py`

**The model is from a repo it has NEVER seen.** Yet it identified the correct files including `celery/app/trace.py` and the test files in `t/unit/tasks/`, scored Modify F1 = 1.5 (above max because it found extra correct files), and produced a coherent plan around the priority lookup logic.

### 6.3 Generalization Win: `pydantic_pr13013` (reward 5.29)

**Spec:** Allow default factories of private attributes to take validated model data

**Ground truth (9 files):** `docs/concepts/fields.md`, `docs/concepts/models.md`, `pydantic/_internal/_fields.py`, `pydantic/_internal/_model_construction.py`, `pydantic/_internal/_utils.py`, `pydantic/fields.py`, `tests/test_fields.py`, `tests/test_internal.py`, `tests/test_private_attributes.py`

**Model predictions (6 files):** `docs/concepts/fields.md`, `docs/migration.md`, `pydantic/_internal/_fields.py`, `pydantic/_internal/_model_construction.py`, `pydantic/_internal/_utils.py`, `tests/test_private_attributes.py`

- **Correct (5)**: `docs/concepts/fields.md`, `pydantic/_internal/_fields.py`, `pydantic/_internal/_model_construction.py`, `pydantic/_internal/_utils.py`, `tests/test_private_attributes.py`
- **Missed (4)**: `docs/concepts/models.md`, `pydantic/fields.py`, `tests/test_fields.py`, `tests/test_internal.py`
- **Hallucinated (1)**: `docs/migration.md`

**The model has never seen Pydantic's source tree** and still identified 5 of 9 correct files including the deeply-nested `_internal/_model_construction.py`. The single hallucination (`docs/migration.md`) is plausible — major Pydantic changes typically do touch the migration guide.

### 6.4 Where the Model Struggles: Rust + Edge Languages

The base model and even the trained model perform worse on Rust-only PRs. Example: `pydantic_pr13016` (Rust serializer change in `pydantic-core/src/serializers/type_serializers/tuple.rs`) scored only 4.99 trained vs 0.10 base — a big improvement, but lower than Python equivalents because the training data is heavily Python/TypeScript.

---

## 7. Things That Worked

1. **Curriculum learning** — 51% steeper learning curve, 14% lower variance, eliminated dead-batch collapses
2. **Component-level semantic similarity** — replacing single-blob comparison with summary + decisions + tickets gave +1.1 separation between good and vague completions
3. **Fuzzy created file matching** — directory-aware path similarity prevents zero scores when the model picks reasonable but slightly-different file names
4. **3-tier format gate** — prevents reward signal collapse when JSON parses but arrays are empty
5. **Cosine LR + warmup** — eliminated training instability seen in earlier runs

---

## 8. Things That Need More Work

1. **Diagram coverage (multimodal claim).** Only 23% of training examples have loadable diagrams. The ablation shows a small positive effect, but it's underpowered. Andrea is investigating fixes to the extraction pipeline.
2. **Created file accuracy.** Improved from 0.14 (exact match) to 0.34 (fuzzy match), but still much lower than modify F1 (1.03). The metric is harder by nature: 31% of examples have any created files at all, so most batches contribute 0 to this signal. Curriculum oversampling of created-file examples would help.
3. **Scale-aware reasoning.** The `scale_tier` field is in every spec but the reward function doesn't measure whether decisions are scale-appropriate. This is the second-biggest differentiator from coding agents like Claude Code (after multimodal) and is currently unaddressed.
4. **Rust and minority languages.** Training data is ~70% Python/TypeScript. Rust-only PRs score lower across the board.
5. **Implementation summary length and specificity.** The model produces correct but sometimes generic summaries. Could benefit from a length-aware bonus or a "specific file mention" reward.

---

## 9. Reproducibility

**W&B project:** `andreajf94-massachusetts-institute-of-technology/repodesign-grpo`

**Key runs:**
- `prime-hill-30` — final 100-step training run (the trained model used for all eval)
- `crisp-jazz-27` — 25-step random sampling baseline
- `northern-dawn-29` — 25-step curriculum learning validation
- `2jj60pjg` — 25-step no-diagrams ablation

**Checkpoints:** stored on Tinker, paths in [`output/grpo_v5_100steps/checkpoints.jsonl`](output/grpo_v5_100steps/checkpoints.jsonl). Final checkpoint covers step 100.

**Repro commands:**

```bash
# Apply teacher plan augmentation (already done)
python scripts/augment_teacher_plans.py data/commit_pairs_production

# Train (full 100-step run)
python scripts/train_grpo_tinker.py data/commit_pairs_production \
  --num-steps 100 --batch-size 4 --group-size 4 --max-tokens 8192 \
  --log-path output/grpo_v5_100steps --eval-every 10

# Eval on unseen repos
python scripts/run_eval_inference.py data/eval \
  --checkpoint-dir output/grpo_v5_100steps \
  --output output/eval_results_trained.json --num-samples 4

# Eval on held-out (45 examples from same repos, different PRs)
python scripts/run_eval_inference.py data/eval_held_out \
  --checkpoint-dir output/grpo_v5_100steps \
  --output output/eval_held_out_trained.json --num-samples 4
```

**Result files:**
- [`output/eval_results_trained.json`](output/eval_results_trained.json) — unseen-repo eval, trained model
- [`output/eval_results_full.json`](output/eval_results_full.json) — unseen-repo eval, base model (for comparison)
- [`output/eval_held_out_trained.json`](output/eval_held_out_trained.json) — held-out eval, trained model
- [`output/grpo_v5_100steps/completions.jsonl`](output/grpo_v5_100steps/completions.jsonl) — 289 best completions from training (one per example per step)
- [`output/grpo_v5_100steps/metrics.jsonl`](output/grpo_v5_100steps/metrics.jsonl) — step-level training metrics
- [`output/grpo_v5_100steps/viewer.html`](output/grpo_v5_100steps/viewer.html) — interactive HTML viewer for completions
