# RepoDesign: Comprehensive Project Analysis

## 1. Project Overview

**RepoDesign** is a multimodal AI agent system that generates **codebase-aware implementation plans** by taking three inputs:
1. A product specification (requirements document)
2. An existing code repository
3. (Optional) Architecture diagrams

The system produces:
- **Architecture decisions** with rationale and affected files
- **Implementation tickets** with real file paths, effort estimates, and dependencies
- **Technology recommendations** scale-appropriate to the project

**Core Innovation**: The model grounds its recommendations in actual repository structure, producing real file paths and respecting project scale constraints (team size, budget, user count).

**Model**: Fine-tuned from `Qwen3-VL-235B-A22B-Instruct` using LoRA (rank 64) via the Tinker SDK with GRPO (Group Relative Policy Optimization).

---

## 2. Project Setup & Key Files

### Dependencies ([pyproject.toml](pyproject.toml))
- **Core**: pydantic, requests, PyGithub, openai, pyyaml, click, rich, python-dotenv
- **Dev**: pytest, pytest-cov
- **Training**: tinker, wandb (Weights & Biases for experiment tracking)
- **Legacy**: anthropic (older LLM provider)

### Python Version
- Requires: Python ≥ 3.10

### Quick Start Commands
```bash
pip install -e ".[dev]"                    # Install in dev mode
pip install -e ".[training]"              # Add training dependencies
python -m pytest tests/ -v                 # Run tests
```

---

## 3. Source Code Structure: `src/repodesign/`

### 3.1 **Schemas** (`schemas/`)
Pydantic data models defining the entire pipeline architecture:

- **`spec.py`**: Product specification schema
  - `ScaleTier`: HOBBY / STARTUP / GROWTH / ENTERPRISE
  - `ScaleConstraints`: expected_users, team_size, budget, timeline
  - `Constraints`: must_use/must_not_use technologies, hard constraints
  - `Spec`: Normalized PRD with functional requirements

- **`repo_ir.py`**: Repository Intermediate Representation (core innovation)
  - `RepoMetadata`: name, URL, language, LOC, contributors, stars, scale_tier
  - `Dependency`: External packages (runtime/dev/optional)
  - `InternalImport`: File-to-file import relationships
  - `APIRoute`: Detected endpoints (Flask/FastAPI/Django/Express)
  - `DataModel`: ORM models (Django/SQLAlchemy/Prisma/Mongoose)
  - `InfraConfig`: Containerization, databases, CI/CD, cloud provider, deployment files
  - `RepoIR`: Aggregated structural representation of codebase

- **`plan.py`**: Implementation plan schema (training target)
  - `ArchitectureDecision`: Decisions across 20 design dimensions (scope, performance, consistency, etc.)
  - `Ticket`: Actionable items with file paths (to_modify, to_create), effort, dependencies
  - `TechnologyChoice`: Category + choice + rationale
  - `ImplementationPlan`: Complete plan with decisions, tickets, and technology choices

### 3.2 **Extractors** (`extractors/`)
Deterministic code analysis pipeline extracting Repo IR from any repository:

- **`directory_analysis.py`**: 
  - Generate directory tree structure
  - Count lines of code by language
  - Identify key directories (src, tests, config, etc.)

- **`dependency_graph.py`**: 
  - Parse pyproject.toml, package.json, Gemfile, etc.
  - Extract external dependencies with versions and types
  - Build internal import graph between files

- **`api_routes.py`**: 
  - Pattern-matching for Flask, FastAPI, Django, Express endpoints
  - Extract route paths, methods, handler files, auth requirements

- **`orm_models.py`**: 
  - Detect Django models, SQLAlchemy classes, Prisma schemas, Mongoose models
  - Extract field types, constraints, relationships

- **`infra_config.py`**: 
  - Parse Dockerfile, docker-compose.yml, k8s manifests, Terraform configs
  - Detect CI/CD systems (GitHub Actions, GitLab CI, CircleCI, etc.)
  - Identify cloud providers, databases, caching layers, message queues

- **`llm_summarizer.py`**: 
  - Uses Claude/GPT/DeepSeek to generate architectural summary
  - One-shot prompt condenses complex codebase structure into text

- **`pipeline.py`**: 
  - Orchestrator that runs all extractors sequentially
  - Produces final RepoIR JSON
  - Collects file manifest for evaluation without needing cloned repo

### 3.3 **Curation** (`curation/`)
GitHub repository scraping and scale classification:
- Fetches repositories by language and star count
- Classifies projects by scale tier (hobby → enterprise)
- Filters for quality and completeness

### 3.4 **Spec Normalizer** (`spec_normalizer/`)
Converts raw product requirement documents (PRDs) into canonical Spec JSON:
- Takes unstructured text or markdown PRDs
- Calls Claude/GPT to extract structure, requirements, constraints
- Outputs validated Spec pydantic model

### 3.5 **Diagrams** (`diagrams/`)
Architecture diagram mining:
- Detects diagram files in repos (SVG, PNG, PlantUML, Mermaid)
- Extracts visual structure
- Grounds diagrams against Repo IR (multimodal fusion)

### 3.6 **Training** (`training/`)
Fine-tuning pipeline using Tinker SDK and GRPO:

- **`data_gen.py`**: 
  - Generates training prompts from Repo IR + Spec
  - Formats as multimodal input for Qwen3-VL model
  - Summarizes repo structure into context window

- **`data_gen_commit_pair.py`**: 
  - Converts real GitHub PRs into training examples
  - Ground truth: modified/created files from diff
  - Spec: extracted from PR title + description
  - Teacher plan: generated from commit pair

- **`reward.py`**: 
  - 7-component reward function (total max: 7.75 points)
    - Format compliance (0.5): Valid JSON structure
    - Format partial (0.25): Near-valid JSON credit
    - Existing file accuracy (1.5): F1 score vs ground-truth modified files
    - Created file accuracy (1.5): Fuzzy F1 of new files (nearest-neighbor matching)
    - Semantic similarity (3.0): Embedding cosine similarity (summary 50%, decisions 30%, tickets 20%)
    - Structural quality (0.5): Ticket/decision counts, T-NNN ID format, dependencies populated
    - Non-empty bonus (0.5): Flat reward if both architecture_decisions and tickets populated
  - Format gate prevents reward collapse: invalid JSON gets only format_partial

- **`tinker_config.py`**: 
  - LoRA configuration for Qwen3-VL-30B fine-tuning
  - Batch size, learning rate, warmup, KL penalty settings

- **`vl_renderer.py`**: 
  - Renders diagrams as images for multimodal input
  - Qwen3VLRenderer integrates diagram images into context

### 3.7 **Evaluation** (`evaluation/`)
Repo Grounding Score (RGS):
- Primary metric: % of file paths in generated plan that exist in target repo
- Fully deterministic, no LLM-as-judge
- Split into existing vs created file accuracy
- No file manifest needed (precomputed during extraction)

---

## 4. Data Organization: `data/`

### 4.1 Repository Lists
- **`repo_list.json`**: Full curated list of open-source repositories for training
- **`repo_list_multimodal.json`**: Repos with diagram assets (for multimodal training)
- **`repo_list_test.json`**: Smaller test set for quick validation

### 4.2 Commit Pairs (Training Data)
Real GitHub PRs converted to training examples:

- **`commit_pairs_production/`**: 232 commit pairs from 25 popular repos
  - `commit_pairs.json`: Metadata (repo, PR number, before/after SHAs, diff files, scale_tier)
  - Subdirectories: `{repo}_{pr_number}/`
    - `repo_ir.json`: Extracted repository IR
    - `teacher_plan.json`: Generated implementation plan (training target)
    - `spec.json`: Extracted specification from PR title + description

- **`commit_pairs_diagrams_only/`**: Subset with only diagram examples

### 4.3 Other Data
- **`pr_diffs/`**: Raw diffs from GitHub PRs (for reference)
- **`eval/`**: Evaluation set examples
- **`repo_research/`**: Research on repository patterns, scale tiers

---

## 5. Scripts Purpose: `scripts/`

### Data Collection & Preparation
1. **`scrape_repos.py`**: Fetch GitHub repos by language/stars, classify scale tier
2. **`fetch_test_repos.py`**: Download test repositories locally
3. **`extract_commit_pairs.py`**: Mine GitHub PRs, convert to commit pairs
4. **`rebuild_commit_pairs_json.py`**: Reconstruct commit_pairs.json metadata
5. **`repair_data_quality.py`**: Fix malformed examples, validate schemas

### Repo IR & Spec Extraction
6. **`extract_repo_ir.py`**: CLI to extract Repo IR from a single repository
7. **`run_extraction_batch.py`**: Batch-process multiple repos, extract all IRs
8. **`normalize_spec.py`**: Convert raw PRD text → canonical Spec JSON

### Training Data Generation
9. **`generate_ir_diagrams.py`**: Mine diagrams from repos, integrate into Repo IR
10. **`generate_teacher_plans.py`**: Generate Spec+RepoIR → ImplementationPlan using teacher model
11. **`generate_file_candidates.py`**: Generate candidate file paths for validation
12. **`batch_generate_candidates.py`**: Batch candidate generation
13. **`augment_teacher_plans.py`**: Augment plans with additional annotations (effort, dependencies)
14. **`generate_training_data.py`**: Orchestrate full pipeline: collect repos → extract IRs → generate specs → generate plans

### Model Training
15. **`train_sft_tinker.py`**: Supervised Fine-Tuning (SFT) stage on Repo IR extraction + Plan generation
16. **`train_grpo_tinker.py`**: GRPO stage with curriculum learning and reward shaping
    - Curriculum: progressive difficulty (easy → medium → hard based on # files)
    - EMA reweighting: examples in "learning zone" get 1.5× sampling weight
    - 100-step run reaches 4.5+ mean reward with 98% format compliance

### Evaluation & Analysis
17. **`run_eval_inference.py`**: Run inference on evaluation set, compute RGS scores
18. **`inspect_data.py`**: Analyze data quality, visualize examples
19. **`mine_diagrams.py`**: Extract architecture diagrams from repos

---

## 6. Training Pipeline & Results

### 6.1 Two-Stage Training

**Stage 1: SFT (Supervised Fine-Tuning)**
- Task: Repo IR extraction (repo files → structured IR)
- Task: Spec + IR → Implementation Plan generation
- Objective: Teach the model the basic input-output format

**Stage 2: GRPO (Group Relative Policy Optimization)**
- Task: Spec + IR → Implementation Plan generation
- Objective: Optimize plan quality using multi-component reward function
- Curriculum: Easy examples (≤3 files) → Medium (4-6) → Hard (≥6 files)
- Progressive curriculum: 60/30/10 → 30/30/40 ratio shift over 50 steps

### 6.2 Curriculum Learning (25-step ablation)

| Metric | Random | Curriculum |
|--------|--------|-----------|
| First-5 avg reward | 2.50 | 2.30 |
| Last-5 avg reward | 3.64 | **4.01** (+10% improvement) |
| Trend (improvement) | +1.14 | **+1.72** (+51% steeper) |
| Std deviation | 0.93 | **0.80** (-14% less variance) |
| Floor (min reward) | 1.05 | **1.76** (eliminated collapses) |

**Result**: Curriculum learning gave steeper improvements, lower variance, and prevented dead-batch collapses.

### 6.3 Full 100-Step GRPO Run (`prime-hill-30`)

| Phase | Steps | Mean Reward | Format Compliance | Modify F1 | Created F1 | Semantic |
|-------|-------|------------|-------------------|-----------|------------|----------|
| Warmup | 0–10 | 2.63 | 30% | 0.32 | 0.01 | 1.85 |
| Transition | 30–50 | 3.91 | 79% | 0.66 | 0.17 | 2.07 |
| Mature | 51–75 | 4.51 | 98% | 0.86 | 0.15 | 2.27 |
| Final | 76–99 | 4.52 | 98% | 0.86 | 0.14 | 2.29 |

**Key Milestones**:
- First 100% format compliance: Step 43
- Sustained ≥90% format compliance: Step 50+
- Peak reward: 5.31 (Step 88)
- First-10 → Last-10 improvement: +1.83 (2.62 → 4.45)
- 35 of last 40 steps at 100% format compliance

### 6.4 Hyperparameters
- LoRA rank: 64 | Batch size: 4 | Group size (G): 4
- Max generation tokens: 8192
- Learning rate: 1e-5 → 1e-6 (cosine decay), 5% warmup
- KL penalty β: 0.1
- Advantage clipping: ±2.0
- Loss: Importance sampling (PPO-style)

---

## 7. Output & Results: `output/`

### Evaluation Results
- **`eval_results.json`**: Evaluation on base model (untrained)
- **`eval_results_trained.json`**: Results after SFT training
- **`eval_results_full.json`**: Results after full GRPO training
- **`eval_held_out_trained.json`**: Held-out test set (trained model)
- **`eval_v7_unseen.json`**: Unseen repository examples
- **`eval_v7_held_out.json`**: Held-out evaluation set

### Training Runs (Weights & Biases experiments)
- **`grpo_v5_100steps/`**: 100-step GRPO baseline
- **`grpo_v7_created_reward/`**: GRPO with improved created-file reward component
- **`grpo_full_diagrams/`**: Multimodal training WITH diagram inputs
- **`grpo_full_no_diagrams/`**: Multimodal training WITHOUT diagram inputs
- **`grpo_diagonly_with/`**: GRPO on diagrams-only subset WITH diagrams
- **`grpo_diagonly_without/`**: GRPO on diagrams-only subset WITHOUT diagrams
- **`grpo_20step/`**: Quick 20-step runs for fast iteration

### Evaluation Metrics in Results
Each evaluation file contains:
- `repo_name`: Test example identifier
- `spec`: Functional requirements extracted from PR
- `ground_truth`: Modified/created files (from actual commit)
- `best_completion`: Generated plan (best of 4 samples)
- `best_rewards`: Component-wise reward breakdown
  - format_compliance, format_partial, existing_file_accuracy, created_file_accuracy, semantic_similarity, structural_quality, total
- `all_rewards`: Rewards for all 4 sampled completions
- `time_seconds`: Generation latency

---

## 8. Documentation & Paper

### `docs/`
- **`neurips_2025.tex`**: LaTeX source for NeurIPS 2025 submission
- **`neurips_2025.sty`**: NeurIPS conference style file
- **`sprint_plan.md`**: Development sprint tracking (tasks, milestones, blockers)

### [REPORT_RESULTS.md](REPORT_RESULTS.md)
Comprehensive midterm report (as of 2026-04-06) covering:
- Training pipeline architecture
- Reward function design (7 components, motivation, gates)
- Curriculum learning strategy and ablation results
- Phase-by-phase performance analysis
- Qualitative plan examples
- Comparative analyses (random vs curriculum, with/without diagrams)

---

## 9. Testing: `tests/`

### `test_extractors.py`
**Unit tests for extraction pipeline** (smoke tests against the RepoDesign repo itself):
- `TestDirectoryAnalysis`:
  - `test_generate_tree()`: Verify directory tree generation includes src/, scripts/
  - `test_count_loc()`: Verify LOC counting works (python language detected)
  - `test_key_directories()`: Identify source code vs test directories
  - `test_full_extract()`: End-to-end extraction produces valid DirectoryInfo

- `TestDependencyGraph`:
  - `test_extract_from_pyproject()`: Parse pyproject.toml, verify pydantic dependency detected

### `test_schemas.py`
**Pydantic schema validation tests**:
- `TestSpec`: Create Spec, validate scale_tier, serialization round-trip
- `TestRepoIR`: Repo IR creation and field validation
- `TestPlan`: Implementation plan schema validation
- Verify serialization/deserialization works correctly (JSON round-trip)

---

## 10. Integration: Workflow & Data Flow

### Full Pipeline Flow

```
┌─ Data Collection
│  ├─ scrape_repos.py → repo_list.json, scale tiers
│  └─ fetch_test_repos.py → clone local copies
│
├─ Repo IR Extraction
│  ├─ extract_repo_ir.py / run_extraction_batch.py
│  ├─ extractors/ (directory_analysis, dependency_graph, api_routes, orm_models, infra_config)
│  ├─ llm_summarizer.py (optional architectural summary)
│  └─ → repo_ir.json files in data/commit_pairs_production/{repo}_{pr}/
│
├─ Spec Generation & Normalization
│  ├─ extract_commit_pairs.py → mine GitHub PRs
│  ├─ normalize_spec.py → PRD text → Spec JSON
│  └─ → spec.json files
│
├─ Training Data Generation
│  ├─ generate_teacher_plans.py → Spec + RepoIR → ImplementationPlan
│  ├─ augment_teacher_plans.py → Add effort, dependencies, scale-awareness
│  ├─ generate_ir_diagrams.py → Mine & integrate diagrams
│  └─ → teacher_plan.json files
│
├─ Training
│  ├─ train_sft_tinker.py → SFT stage (format learning)
│  ├─ train_grpo_tinker.py → GRPO stage (quality optimization)
│  │  ├─ StratifiedSampler (curriculum: easy/medium/hard)
│  │  ├─ reward.py (7-component reward function)
│  │  ├─ tracking to W&B (Weights & Biases)
│  │  └─ checkpoints saved locally
│  └─ → Trained LoRA weights
│
├─ Evaluation
│  ├─ run_eval_inference.py
│  ├─ evaluation/repo_grounding_score.py (RGS metric)
│  ├─ Compute % of file paths that exist in target repos
│  └─ → eval_results_*.json
│
└─ Analysis & Publication
   ├─ inspect_data.py → Data quality visualization
   ├─ REPORT_RESULTS.md → Comprehensive results summary
   └─ docs/neurips_2025.tex → Academic paper
```

### Data-to-Model Flow

```
GitHub Repository
    ↓
[extract_repo_ir.py]
    ↓
Repo IR (structured architecture representation)
    ↓ + Real GitHub PR
[generate_teacher_plans.py]
    ↓
Spec + RepoIR + Ground-truth modified/created files
    ↓
[data_gen_commit_pair.py]
    ↓
Training Example: {prompt: Spec+RepoIR, completion: ImplementationPlan, ground_truth: files}
    ↓
[train_sft_tinker.py / train_grpo_tinker.py]
    ↓
Fine-tuned Qwen3-VL-30B model (LoRA)
    ↓
[run_eval_inference.py]
    ↓
RGS Score (% file paths correct)
```

---

## 11. Key Design Innovations

### 1. **Repo IR (Intermediate Representation)**
- **Deterministic extraction**: No LLM needed (extractors are pattern-based)
- **Structural capture**: APIs, ORM models, dependencies, infrastructure
- **Scale-aware**: Includes project metadata for scale-tier reasoning

### 2. **Multimodal Fusion**
- Integrates text (Repo IR + Spec) with architecture diagrams
- Grounds visual architecture against code structure
- Enables scale-contrastive learning (small vs large projects)

### 3. **Repo Grounding Score (RGS)**
- **Metric**: % of generated file paths that actually exist in repo
- **Deterministic**: No LLM judgment (binary ground truth)
- **Immediate feedback**: File manifest precomputed; no runtime repo access needed

### 4. **7-Component Reward Function**
- Prevents reward collapse with format gates
- Combines structure (format, schema) + semantics (embeddings) + accuracy (F1 on paths)
- Weights content (semantic 3.0) > accuracy (1.5+1.5) > structure (0.5+0.5)

### 5. **Curriculum Learning**
- Stratified sampler by ground-truth difficulty (# files modified/created)
- Progressive ratio shift: early training on easy examples, later on hard
- EMA-based per-example weighting: focus on "learning zone" examples
- **Result**: 51% steeper improvement curve vs random batching

---

## 12. Team & Roles

- **Andrea Jimenez Fernandez** — Repo IR extraction, data pipeline
- **Cerine Hamida** — Multimodal integration, training pipeline
- **Kevin Power** — Scale-aware reasoning, evaluation framework

---

## 13. Project Maturity & Status

**Stage**: Mid-stage research project, preparing for NeurIPS 2025 submission

**Completed**:
- ✅ Repo IR extraction pipeline (deterministic)
- ✅ Training data generation from 232 real GitHub PRs
- ✅ SFT + GRPO training with curriculum learning
- ✅ Comprehensive reward function with format gating
- ✅ Multimodal extension (Qwen3-VL + diagrams)
- ✅ Evaluation framework (RGS metric)
- ✅ Midterm results & ablations documented

**In Progress**:
- 📊 Extended evaluation on held-out + unseen repositories
- 📊 Diagram-assisted training (comparing with/without diagrams)
- 📝 NeurIPS 2025 paper draft (docs/neurips_2025.tex)

**Future**:
- Deploy inference API
- Extend to additional languages (Java, Go, Rust)
- Integrate LM cascade for long-context repositories
- User study validation with real development teams

---

## Summary

RepoDesign is a sophisticated multimodal ML system that bridges the gap between AI planning and real codebase complexity. By extracting deterministic Repo IRs and grounding recommendations in actual file paths, it demonstrates how to build AI agents that produce executable, repository-aware implementation plans. The project combines solid software engineering (extraction, validation), thoughtful ML design (curriculum, reward shaping), and rigorous evaluation (RGS metric, held-out tests) to tackle the challenging problem of scale-aware system design assistance.
