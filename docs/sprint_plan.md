# Sprint Plan — Week of April 7

## Individual Tasks

### Cerine
- Fix the **created files reward** component in GRPO training
- Investigate the `implementation_summary` field — determine whether it can support more targeted semantic matching
- If needed, regenerate teacher examples with additional structured fields to improve reward signal granularity

### Kevin
- Generate **visual representations of RepoIR** fields (api_routes, data_models) as Mermaid → PNG diagrams
- Integrate generated images into the multimodal training pipeline via `load_diagram_images()` and `Qwen3VLRenderer`
- Evaluate diagram quality across the dataset and iterate on rendering logic

### Andrea
- Draft the **mid-term report** based on current training data, pipeline architecture, and methodology
- Leave placeholders for preliminary benchmark results (to be filled once the benchmark dataset is ready)

## Shared Tasks (All)

### Benchmark Dataset
- **Before Tuesday:** assemble a small benchmarking subset from held-out commit pairs
- Design the benchmark to evaluate plan quality: file accuracy (RGS), semantic similarity (TF-IDF), and structural completeness
- Run the benchmark against:
  1. Our latest fine-tuned model
  2. Other open-source models (baseline comparison)
- Consider using **Kaggle Datasets** to host the benchmark (free tier available) to reduce API costs

### Training Hygiene
- Audit decoding settings: confirm we are using **greedy decoding** (or controlled sampling) during evaluation to ensure deterministic, reproducible benchmark results
- Review non-determinism sources in both the training loop and benchmark pipeline

## Benchmarking Strategy

**Goal:** Demonstrate that our open-source model excels at codebase-aware system design planning — free for anyone to use.

**Comparisons:**

| Category | Purpose |
|---|---|
| Open-source models (Qwen, LLaMA, Mistral, etc.) | Show our fine-tuned model outperforms general-purpose OS models on system design tasks |
| Smaller proprietary models (GPT-4o-mini, Claude Haiku, etc.) | Demonstrate competitiveness with commercial offerings |
| Text vs. image modality | Ablation study — feed api_routes and data_models as raw JSON text vs. rendered diagram images and measure which provides greater improvement to plan quality |

The text-vs-image ablation is particularly valuable: it directly justifies the multimodal approach by quantifying the benefit of visual representations over equivalent textual input.
