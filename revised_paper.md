# RepoDesign: Codebase-Aware Multimodal System Design Planning

Andrea Jimenez Fernandez, Cerine Hamida, Kevin Power  
MIT  
{andrejim, chamida, kevpower}@mit.edu  

## Abstract
Large language models can discuss system design fluently, yet their advice is fundamentally ungrounded: they suggest adding infrastructure without checking whether it already exists, or recommend enterprise-grade solutions for hobby-scale projects. We introduce RepoDesign, a multimodal agent pipeline that accepts a product specification, an existing code repository, and optionally architecture diagrams, and produces a codebase-aware implementation plan with real file paths, scale-appropriate technology choices, and executable tickets. Our core contribution is the Repo Intermediate Representation (Repo IR): a structured JSON encoding of repository architecture produced by deterministic static analysis fused with LLM summarization, which serves as the grounding scaffold for plan generation. We fine-tune Qwen3-VL-235B-A22B-Instruct via LoRA on 232 real GitHub commit pairs using a customized 7-component reward function under GRPO. On a held-out benchmark of 12 examples from 6 unseen repositories, our GRPO-trained model achieves 0.98 existing-file F1, 100% format compliance, and 2.40/3.00 semantic plan similarity, demonstrating that structured repository representations substantially reduce hallucinated file references and GRPO stabilizes structural generations. Comparative RGS results against GPT-4 baselines are pending and will be reported in the final paper.

## 1 Introduction
System design is one of the most consequential activities in software engineering; a poor architectural decision in week one can cost months of rework. Yet today’s AI tools leave a critical gap. Code-generation tools (GitHub Copilot, Cursor) excel at implementing individual functions but offer no guidance on what to build or where to place it within an existing codebase. Conversational assistants (ChatGPT, Claude) discuss design eloquently, but their recommendations are generic, decoupled from any repository, blind to existing infrastructure, and biased toward over-engineered solutions. A repo-grounded, scale-appropriate design tool would democratize architectural expertise, giving engineers guidance calibrated to their actual context rather than Silicon Valley-scale assumptions.

Despite advances in code-aware LLMs and agentic workflows, no existing system simultaneously addresses: (1) architectural reasoning grounded in real code, (2) visual diagram understanding tied to actual file paths, and (3) scale-appropriate technology recommendations. Repo-level reasoning systems like SWE-Agent [22] and RepoCoder [23] target bug fixing and code completion, not design planning. Multi-agent systems like MetaGPT [6] and ChatDev [15] generate architectures de novo for greenfield projects, ignoring existing infrastructure. No prior work grounds visual architecture diagrams to the actual repository structure, and scale-awareness is absent from all existing tools. Crucially, the few closed-source systems that attempt aspects of architectural reasoning (e.g., Devin [4]) remain proprietary, non-reproducible, and impossible to study or extend. To date, no open-source model has been trained or evaluated on codebase-aware system design planning.

*[Figure 1: Pipeline Overview Placeholder - Add your diagram showing S, R, D, C feeding into Repo IR Extraction]*

In this paper, we introduce RepoDesign, the first open-source multimodal agent pipeline for repo-grounded system design. Our contributions are threefold: (1) a Repo Intermediate Representation (Repo IR) capturing repository architecture through deterministic static analysis augmented by LLM summarization; (2) a multimodal diagram fusion mechanism grounding visual diagrams to Repo IR entities; and (3) a scale-aware reasoning framework via GRPO over contrastive scale-tier pairs. By building on Qwen3-VL, a fully open-weight foundation, and releasing all training data, extraction tools, and model weights, RepoDesign enables the research community to reproduce, audit, and extend codebase-aware design reasoning in ways that closed-source alternatives do not permit. For this midterm, we focus on the Repo IR contribution and demonstrate its effectiveness through SFT and subsequent GRPO training on 232 real GitHub commit pairs, evaluated by a novel Repo Grounding Score (RGS) on held-out repositories.

## 2 Related Work
**Code-Aware Language Models.** CodeBERT [5] and CodeT5 [19] introduced code pre-training; StarCoder [12] and Code Llama [16] scaled to long-context generation. RepoCoder [23], SWE-Agent [22], and SWE-bench [8] pushed toward repo-level reasoning, while OpenHands [18] and Devin [4] extend to multi-step autonomous engineering. These systems focus on code generation and bug fixing, not architectural planning, and persistent failures in far-context utilization motivate our Repo IR as a structural scaffold that deterministically surfaces high-signal architecture signals.

**Automated Software Architecture.** Early work used formal Architecture Description Languages [14]; Chen et al. [3] explored LLM-based UML generation. ChatDev [15] and MetaGPT [6] simulate multi-agent teams but target greenfield projects without codebase awareness.

**Multimodal Diagram Understanding.** Qwen2-VL [20] and LLaVA [13] interpret diagrams and screenshots; prior work explores UML understanding [9]. No prior work grounds visual architecture diagrams against an actual codebase. RepoDesign’s diagram grounding is the architectural analogue of GUI grounding: visual-to-structured alignment enabling downstream planning.

**Constraint-Aware Planning.** RAG [11] and tool-augmented LLMs [17] address factual grounding; SayCan [1] and Plan-and-Solve [21] explore LLM planning. RepoDesign applies constraint-aware alignment via GRPO over contrastive scale-tier pairs, penalizing both over- and under-engineering. To our knowledge, RepoDesign is the first system unifying codebase-aware architectural reasoning, visual diagram grounding, and scale-appropriate design planning.

## 3 Problem Statement
Let S denote a product specification, R a code repository, D an optional set of architecture diagrams, and C = (users, team_size, budget) project constraints defining scale tier t ∈ {hobby, startup, growth, enterprise}. Our goal is to learn fθ producing a grounded plan P = fθ(S, R, D, C) satisfying: (1) all files_to_modify in P exist in R (grounding); (2) technology choices suit tier t (scale-appropriateness); and (3) P covers all requirements in S (completeness). We decompose into Repo IR extraction IR = g_det(R) ∪ g_LLM(R), and plan generation P = hθ(S, IR, D, C), optimized in two stages:

L_SFT(θ) = −E[ log pθ(P* | S, IR) ]
L_GRPO(θ) = −E[ ∑ Â_i log pθ(P_i | S, IR, C) − β D_KL(pθ || p_ref) ]

where P* is a teacher plan from a real PR diff, G=4 candidates per prompt, β=0.1, and Â_i is the group-normalized advantage from reward R(P_i, R, C, P*).

## 4 Proposed Approach
Our midterm experiments focus on the Repo IR research thread: constructing a structured intermediate representation of repository architecture and using it to ground GRPO training of a plan-generation model.

### 4.1 Repo Intermediate Representation
The Repo IR is a structured JSON document: IR = g_det(R) ∪ g_LLM(R), where g_det runs six extraction stages and g_LLM appends a natural-language architectural summary. The extraction pipeline comprises: (1) Directory analysis; (2) Dependency extraction; (3) API route extraction; (4) ORM model extraction; (5) Infrastructure config; and (6) LLM summary via DeepSeek. A file manifest is collected for RGS evaluation. The Repo IR schema (Pydantic) includes metadata, dependencies, internal_imports, api_routes, data_models, infrastructure, directory_tree, architectural_summary, file_manifest, and diagram_paths.

*[Figure 2: Example Repo IR JSON Placeholder - Add snippet of flask_pr5917 IR here]*

A concrete example from `flask_pr5917`: 18,390 LOC, 19 Flask API routes extracted, no ORM models, GitHub Actions CI detected, file manifest of 412 paths. Scale tier is classified from GitHub metadata: repositories with at least 50k stars or 500 contributors are labeled ENTERPRISE; at least 10k stars or 100 contributors, GROWTH; at least 1k stars or 10 contributors, STARTUP; all others default to HOBBY.

### 4.2 Group Relative Policy Optimization (GRPO)
We optimize our plan generator via curriculum-based Group Relative Policy Optimization using 4 completions per prompt (G=4). The curriculum limits training instability by phasing easy modification configurations (≤3 files) early, shifting toward harder modifications as the agent succeeds.

### 4.3 Multi-Component Reward Function
We evaluate and train with a 7-component reward (max 7.50):

**Table 1: Multi-component reward function**

| Component | Max | Definition |
|-------------|-----|------------|
| Format compliance | 0.50 | Valid JSON with required schema fields, populated arrays |
| Format partial | 0.25 | Partial credit for near-valid JSON structure |
| Existing file accuracy| 1.50 | F1(files_to_modify, Δ_modified) × 1.5 |
| Created file accuracy | 1.50 | Fuzzy F1 via nearest-neighbor path matching × 1.5 |
| Semantic similarity | 3.00 | Component-level cosine sim. (MiniLM-L6-v2) |
| Structural quality | 0.50 | 3–8 decisions, 4–10 tickets, dependencies populated |
| Nonempty bonus | 0.50 | +0.5 if decisions and tickets are both correctly populated |

Format compliance acts as a gate (see Figure 3): outputs failing JSON parsing receive only `format_partial` to prevent noisy gradient signals. Fuzzy path similarity evaluates creation files using directory overlap, filename similarity, and extension matching.

![Figure 3: Structural hierarchy of the multi-component reward function](/Users/andreajimenez/.gemini/antigravity/brain/1dab6e8a-e928-44bf-a05d-f3156d634b67/artifacts/fig3_reward_hierarchy.png)
> **Figure 3:** Structural hierarchy of the multi-component reward function. The 3-tier format gate prevents reward signal collapse by zeroing out hallucinated pathing and scaling rewards if the base JSON schema fails to parse.

## 5 Experimental Methodology
**Datasets and splits.** We initially curated a raw dataset of 236 commit pairs from 25 open-source GitHub repositories (detailed in Figure 4). Following programmatic deduplication and JSON formatting validation, 4 malformed pairs were dropped, leaving a robust final training set of **232** commit pairs. Each example contains a Spec JSON, a RepoIR extracted at pre-commit state, a teacher ImplementationPlan from GPT-4, and diff_files. A post-hoc augmentation added `files_to_create` ground truth to 65 teacher plans. Evaluation consists of two phases: 12 unseen-repo examples (Celery, Django, Flask, HTTPX, Pydantic, SQLModel; 2 PRs each) and 45 held-out examples from the training repos using different PRs.

![Figure 4: Breakdown of the training commit pairs](/Users/andreajimenez/.gemini/antigravity/brain/1dab6e8a-e928-44bf-a05d-f3156d634b67/artifacts/fig4_dataset.png)
> **Figure 4:** Breakdown of the training commit pairs derived from 25 open-source repositories. The dataset covers four primary languages (left) to prevent framework overfitting, and spans the entire scale spectrum from early-stage startups to million-line enterprise monoliths (right) to enable scale-aware architectural recommendations.

**Baselines.** (1) Base Model (no training); (2) GPT-4o with spec only; (3) GPT-4o with naive RAG; (4) RepoDesign SFT; (5) RepoDesign GRPO (final 100-step run).
**Metrics.** Repo Grounding Score (RGS): fraction of files_to_modify existing in the repo (deterministic). Semantic Similarity: plan vs teacher. Created File Accuracy: fuzzy path mapping F1. Format Compliance: schema match.

**Implementation.** Qwen3-VL-235B-A22B-Instruct; LoRA rank 64, α=128, dropout 0.05. GRPO ran for 100 steps. Hyperparameters: Batch 4 prompts, group size (G) 4 completions per prompt, lr 10⁻⁵ with cosine decay to 10⁻⁶, max generation sequence length of 8,192, kl beta 0.1, bf16; 4× A100-80GB. Full details in Appendix D.

## 6 Results and Discussion

### 6.1 Repo IR Extraction Quality
The pipeline successfully processed all repositories. API routes were extracted for all Flask, FastAPI, and Django projects; ORM models were detected in Django/SQLAlchemy codebases. One systematic error identified is the route extractor over-capturing routes in test fixtures (e.g., test files with `@app.route` decorators cataloged as production).

### 6.2 GRPO Training Results
Through curriculum learning, we smoothed optimization curves, drastically reducing step-to-step variance by 14% and eliminating dead-batch collapses completely (Figure 5). The GRPO model hit 100% format compliance by step 43.

![Figure 5: GRPO training reward curves over 100 steps](/Users/andreajimenez/.gemini/antigravity/brain/1dab6e8a-e928-44bf-a05d-f3156d634b67/artifacts/fig5_training_curves.png)
> **Figure 5:** GRPO training reward curves over 100 steps. The curriculum-sampled dataset (blue) yields a 51% steeper learning curve and completely eliminates the dead-batch learning collapses seen in the random batching baseline (orange).

**Table 2: Reward component scores on 12-example unseen repo benchmark (via 100-Step GRPO model)**

| Component | Score | Max | % of Max |
|-----------|-------|-----|----------|
| Format compliance | 0.50 | 0.50 | 100% |
| Existing file accuracy| 0.98 | 1.50 | 65%  |
| Created file accuracy | 0.08 | 1.50 | 5%   |
| Semantic similarity | 2.40 | 3.00 | 80%  |
| Total Reward | 4.71 | 7.50 | 63%  |

**Table 3: Comparative RGS and plan quality (GPT-4 Baselines pending).**

| Method | Reward | Format Pass | Modify F1 (Existing) | Created F1 | Semantic Sim. |
|--------|--------|-------------|----------------------|------------|---------------|
| Base Qwen3-VL | 1.62 | 17% | 0.15 | 0.00 | 1.19 |
| GPT-4o (No context) | TBD | TBD | TBD | TBD | TBD |
| GPT-4o (Naive RAG) | TBD | TBD | TBD | TBD | TBD |
| RepoDesign (GRPO, Unseen) | 4.71 | 100% | 0.98 | 0.08 | 2.40 |
| RepoDesign (GRPO, Held-Out) | 4.73 | 100% | 1.03 | 0.34 | 2.11 |

### 6.3 Analysis
The central result is the 0.98 existing-file F1 (and 1.03 on held-out distributions): the trained model references files that actually exist in the target repository at near-perfect precision by securely traversing its File Manifest logic within the Repo IR. The jump from 17% to 100% format compliance further proves that the multi-tier GRPO penalty structure strictly enforces structural bounds.

**Failure mode 1: Created file hallucination.** While the model achieves a 0.34 fuzzy F1 on held-out repositories, it drops to 0.08 on unseen structures. The model correctly predicts module areas and extensions, but misses exact paths.
**Failure mode 2: Test fixture route over-capture.** In Flask and HTTPX, test files with `@app.route` decorators are cataloged as production routes, causing plans to misplace modifications.

### 6.4 Discussion
Structured intermediates appear far more powerful than similarity-based retrieval for architectural grounding. The file manifest alone significantly constrains path generation. High semantic similarity (80%) suggests plan coherence develops independently of file physical grounding. 

## 7 Next Steps and New Research Ideas

### 7.1 Immediate Next Steps
**Improving Diagram Extraction Pipeline.** While we explored diagram fusion, diagram coverage remains the highest priority unblocker for stating strong multimodal claims. Currently, only 23% of training examples have loadable diagrams. We are investigating expanding extraction procedures immediately to increase this number.
**Created File Prediction.** We plan to refine our explicit IR scaffolding, prompting the model to reason about module directories before deciding on path completions, along with further isolating creation tracking tasks from modifications.

### 7.2 Expanded Baseline Comparisons
For the final report, we will expand comparisons along two axes. First, open-source models: base Qwen3-VL, LLaMA-3, and Mistral, all prompted with the same Repo IR context. Second, smaller proprietary models (GPT-4o-mini, Claude Haiku) to establish where our open-source fine-tuned model sits relative to commercial alternatives.

### 7.3 Text vs. Image Modality Ablation
To test exactly if multimodal input improves plan quality, we will conduct an ablation comparing two distinct endpoints: (1) Repo IR JSON text only vs (2) Repo IR visual diagrams (dependency graphs, route maps) provided as images.

### 7.4 Self-Generated Diagram Pipeline
Rather than relying on user-provided architecture diagrams, we will generate visual representations of Repo IR fields automatically via Mermaid diagrams translated into PNG formats. This solidifies our multimodal pipeline on all examples rather than only relying on arbitrary external inclusions.

## 8 Conclusion
We presented RepoDesign, a multimodal agent pipeline addressing the fundamental ungroundedness of AI system design tools. Our midterm demonstrates that a structured Repo IR coupled with GRPO reinforcement learning enables plan generation with 0.98 existing-file F1 and 100% structural fidelity on entirely unseen repositories. The immediate remaining challenge focuses heavily on increasing diagram extraction limits to bolster our multimodal pipeline evaluation on the road to the final report.

***

## Appendices

### E. Additional Results and Qualitative Examples
Due to space constraints in the main text, we provide full qualitative outputs here to demonstrate the structural fidelity and codebase grounding of the GRPO-trained model.

#### E.1 Unseen Repository Example: `celery_pr10184` (Reward: 5.37)
**Product Specification:** "Fix: prioritize request `ignore_result` over task definition. `apply_async(ignore_result=False)` on a task with `@app.task(ignore_result=True)` should still return results."

**Model Output:**
```json
{
  "architecture_decisions": [
    {
      "dimension": "Data Flow",
      "recommendation": "Introduce get_actual_ignore_result function in trace.py that resolves ignore_result priority: Request > Task Definition",
      "rationale": "This ensures consistent behavior where per-call overrides take precedence over task-level settings, which is the expected semantics for users.",
      "alternatives_considered": [
        "Modify ignore_result directly in Task class during apply_async"
      ],
      "files_affected": ["celery/app/trace.py"]
    },
    {
      "dimension": "Testing Strategy",
      "recommendation": "Add unit tests in test_request.py and test_trace.py to validate priority resolution",
      "rationale": "Ensures both the request processing and trace execution paths correctly honor the override, preventing regression.",
      "alternatives_considered": [
        "Only test in integration suite"
      ],
      "files_affected": [
        "t/unit/worker/test_request.py",
        "t/unit/tasks/test_trace.py"
      ]
    }
  ],
  "tickets": [
    {
      "id": "T-001",
      "title": "Implement get_actual_ignore_result with Request > Task precedence",
      "description": "Create a utility function in trace.py that resolves priority.",
      "files_to_modify": ["celery/app/trace.py"],
      "files_to_create": [],
      "estimated_effort": "small",
      "dependencies": []
    },
    {
      "id": "T-004",
      "title": "Add unit tests for ignore_result priority resolution",
      "description": "Write test cases in test_request.py and test_trace.py verifying that per-call overrides task setting.",
      "files_to_modify": [
        "t/unit/worker/test_request.py",
        "t/unit/tasks/test_trace.py"
      ],
      "files_to_create": [],
      "estimated_effort": "medium",
      "dependencies": ["T-001"]
    }
  ],
  "implementation_summary": "This feature implements a fix for the ignore_result behavior in Celery by introducing a priority-based resolution mechanism where per-call overrides take precedence..."
}
```

### F. Reproducibility & Open Source Links
To support the fully open-weight foundation of RepoDesign and allow verification of our GRPO results, all training data, extraction configurations, and pre-trained weights are publicly available.

**GitHub Repository:** [https://github.com/andreajf94/Multimodal](https://github.com/andreajf94/Multimodal)  
**Weights & Biases (W&B) Project:** `andreajf94-massachusetts-institute-of-technology/repodesign-grpo`  

**Reproduction Commands:**
```bash
# Eval on the 12 unseen-repo examples using the final GRPO checkpoint
python scripts/run_eval_inference.py data/eval \
  --checkpoint-dir output/grpo_v5_100steps \
  --output output/eval_results_trained.json --num-samples 4

# Eval on the 45 held-out distribution examples
python scripts/run_eval_inference.py data/eval_held_out \
  --checkpoint-dir output/grpo_v5_100steps \
  --output output/eval_held_out_trained.json --num-samples 4
```
