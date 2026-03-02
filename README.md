# RepoDesign: Codebase-Aware Multimodal System Design Planning

A multimodal agent pipeline that accepts a product specification, an existing code repository, and (optionally) architecture diagrams, and produces a **codebase-aware implementation plan** with real file paths, scale-appropriate technology choices, and executable ticket descriptions.

## Core Contributions

1. **Repo IR** — A structured Intermediate Representation capturing repository architecture through deterministic code analysis + LLM summarization
2. **Multimodal Diagram Fusion** — Grounding visual architecture diagrams against the Repo IR
3. **Scale-Aware Reasoning** — Conditioning recommendations on project constraints (team size, user count, budget) via GRPO

## Project Structure

```
Multimodal/
├── src/repodesign/             # Core Python package
│   ├── schemas/                # Pydantic models (Spec, RepoIR, Plan)
│   ├── extractors/             # Repo IR extraction pipeline
│   │   ├── directory_analysis.py
│   │   ├── dependency_graph.py
│   │   ├── api_routes.py       # Flask/FastAPI/Django/Express
│   │   ├── orm_models.py       # Django/SQLAlchemy/Prisma
│   │   ├── infra_config.py     # Docker/K8s/Terraform/CI
│   │   ├── llm_summarizer.py
│   │   └── pipeline.py         # Orchestrator
│   ├── curation/               # GitHub repo scraping + scale classification
│   ├── spec_normalizer/        # PRD → canonical Spec JSON
│   ├── diagrams/               # Diagram mining from repos
│   ├── evaluation/             # Repo Grounding Score (RGS)
│   └── training/               # Tinker/LoRA config for Qwen3-VL-235B
├── scripts/                    # CLI entry points
├── tests/                      # Test suite
└── data/                       # Data directory (gitignored)
```

## Setup

```bash
# Install in editable mode
pip install -e ".[dev]"
```

## Quick Start

### Extract Repo IR from any repository
```bash
python scripts/extract_repo_ir.py /path/to/repo --skip-llm
python scripts/extract_repo_ir.py /path/to/repo --url https://github.com/user/repo --stars 500
```

### Scrape GitHub repos for training data
```bash
export GITHUB_TOKEN=ghp_your_token
python scripts/scrape_repos.py --per-language 50
```

### Run batch extraction
```bash
python scripts/run_extraction_batch.py data/repo_list.json --skip-llm --limit 10
```

### Mine architecture diagrams
```bash
python scripts/mine_diagrams.py /path/to/repo1 /path/to/repo2
```

### Normalize a PRD
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/normalize_spec.py prd.txt -o data/specs/spec-001.json
```

## Run Tests
```bash
python -m pytest tests/ -v
```

## Training Pipeline

**Model:** Qwen3-VL-235B-A22B-Instruct via full LoRA (Tinker)
- **Stage 1 (SFT):** Repo→IR extraction + Spec+IR→Plan generation
- **Stage 2 (GRPO):** Scale-contrastive preference pairs

See `src/repodesign/training/tinker_config.py` for configuration.

## Evaluation

**Primary metric:** Repo Grounding Score (RGS) — % of file paths in the generated plan that actually exist in the target repository. Fully deterministic, no LLM-as-judge.

## Team

- **Andrea Jimenez Fernandez** — Repo IR & Data Pipeline
- **Cerine Hamida** — Multimodal & Training
- **Kevin Power** — Scale Reasoning & Evaluation
