# Systems Design Multimodal Model — Budget Fine-Tuning Pipeline

## Overview
This is the complete, minimal pipeline to fine-tune Qwen2-VL-7B-Instruct into a systems design assistant. Total cost: ~$150-250.

## Directory Structure
```
sysdesign-finetune/
├── README.md                  ← You are here
├── 01_generate_diagrams.py    ← Generates synthetic architecture diagrams
├── 02_generate_conversations.py ← Generates training conversations using Claude/GPT-4 API
├── 03_prepare_dataset.py      ← Converts raw data into training format
├── 04_train.py                ← QLoRA fine-tuning with Unsloth
├── 05_merge_and_export.py     ← Merges LoRA adapter and exports to GGUF
├── 06_evaluate.py             ← Runs evaluation on test prompts
├── 07_inference.py            ← Interactive inference script
├── prompts/
│   ├── system_design_prompts.json    ← 50 seed prompts for data generation
│   └── eval_prompts.json             ← 30 held-out test prompts
├── data/
│   ├── raw/                   ← Raw generated conversations
│   ├── diagrams/              ← Generated architecture diagrams
│   └── train.jsonl            ← Final training dataset
└── output/
    ├── checkpoints/           ← Training checkpoints
    ├── merged/                ← Merged full model
    └── gguf/                  ← Quantized GGUF for deployment
```

## Prerequisites

### Local machine (for data generation)
```bash
pip install anthropic openai diagrams Pillow matplotlib
```

### Training machine (rent a GPU — Vast.ai or RunPod)
```bash
# On a fresh Ubuntu + CUDA machine with A100 40/80GB:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install qwen-vl-utils
```

### Deployment machine (your laptop or a cheap VPS)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

## Pipeline Steps

### Step 1: Generate training data (your laptop, ~20-40 hours of work)
```bash
python 01_generate_diagrams.py          # ~30 min, generates 500 diagrams
python 02_generate_conversations.py     # ~2-4 hours, costs ~$50-100 in API fees
python 03_prepare_dataset.py            # ~5 min, formats everything
```

### Step 2: Train (rented GPU, ~4-8 hours)
```bash
# Upload data/ folder to GPU machine
python 04_train.py                      # ~4-8 hours on A100
python 05_merge_and_export.py           # ~20 min
```

### Step 3: Deploy (your laptop or VPS)
```bash
# Download the GGUF file from GPU machine
ollama create sysdesign -f Modelfile
ollama run sysdesign
```

### Step 4: Evaluate and iterate
```bash
python 06_evaluate.py                   # Test against held-out prompts
# Review results, improve training data, retrain
```
