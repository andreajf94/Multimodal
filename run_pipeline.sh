#!/bin/bash
# ═══════════════════════════════════════════════════════════
# run_pipeline.sh — Full pipeline automation
#
# This script runs all steps in sequence.
# Edit the variables below to customize.
# ═══════════════════════════════════════════════════════════

set -e  # Exit on error

# ─── Configuration ───
PROVIDER="anthropic"          # "anthropic" or "openai"
VARIANTS_PER_PROMPT=20        # Conversations per seed prompt (20 * 50 = 1000)
MAX_PROMPTS=50                # How many of the 50 seed prompts to use
TRAIN_EPOCHS=3
LORA_RANK=32
LEARNING_RATE=2e-4
BATCH_SIZE=2
GRAD_ACCUM=16
MAX_SEQ_LEN=4096
QUANT_TYPE="q4_k_m"

echo "═══════════════════════════════════════════"
echo "  Systems Design Model — Training Pipeline"
echo "═══════════════════════════════════════════"
echo ""
echo "  Provider: $PROVIDER"
echo "  Conversations: $((VARIANTS_PER_PROMPT * MAX_PROMPTS))"
echo "  Epochs: $TRAIN_EPOCHS"
echo "  LoRA rank: $LORA_RANK"
echo ""

# ─── Step 1: Generate diagrams ───
echo ""
echo "══════ STEP 1/6: Generate Diagrams ══════"
echo ""
if [ -f "data/raw/diagram_descriptions.jsonl" ]; then
    echo "Diagrams already exist. Skipping."
else
    pip install diagrams Pillow 2>/dev/null
    sudo apt-get install -y graphviz 2>/dev/null || true
    python 01_generate_diagrams.py
fi

# ─── Step 2: Generate conversations ───
echo ""
echo "══════ STEP 2/6: Generate Conversations ══════"
echo ""
CONV_COUNT=$(find data/raw/conversations -name "*.json" 2>/dev/null | wc -l)
TARGET=$((VARIANTS_PER_PROMPT * MAX_PROMPTS))

if [ "$CONV_COUNT" -ge "$TARGET" ]; then
    echo "$CONV_COUNT conversations exist (target: $TARGET). Skipping."
else
    if [ "$PROVIDER" = "anthropic" ]; then
        pip install anthropic 2>/dev/null
        [ -z "$ANTHROPIC_API_KEY" ] && echo "ERROR: Set ANTHROPIC_API_KEY" && exit 1
    else
        pip install openai 2>/dev/null
        [ -z "$OPENAI_API_KEY" ] && echo "ERROR: Set OPENAI_API_KEY" && exit 1
    fi
    python 02_generate_conversations.py \
        --provider "$PROVIDER" \
        --count "$VARIANTS_PER_PROMPT" \
        --max-prompts "$MAX_PROMPTS"
fi

# ─── Step 3: Prepare dataset ───
echo ""
echo "══════ STEP 3/6: Prepare Dataset ══════"
echo ""
python 03_prepare_dataset.py

TRAIN_COUNT=$(wc -l < data/train.jsonl)
echo "Training examples: $TRAIN_COUNT"

# ─── Step 4: Train ───
echo ""
echo "══════ STEP 4/6: Training ══════"
echo ""
echo "This is the expensive part. Estimated time: 4-8 hours on A100."
echo ""

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: No GPU detected. Training will be very slow."
    echo "Rent a GPU from Vast.ai or RunPod (\$0.50-1.00/hr for A100)."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

python 04_train.py \
    --epochs "$TRAIN_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lora-rank "$LORA_RANK" \
    --max-seq-len "$MAX_SEQ_LEN"

# ─── Step 5: Export ───
echo ""
echo "══════ STEP 5/6: Merge & Export ══════"
echo ""
python 05_merge_and_export.py --quant "$QUANT_TYPE"

# ─── Step 6: Evaluate ───
echo ""
echo "══════ STEP 6/6: Evaluate ══════"
echo ""

# Check if Ollama is available for eval
if command -v ollama &> /dev/null; then
    GGUF_FILE=$(find output/gguf -name "*.gguf" | head -1)
    if [ -n "$GGUF_FILE" ]; then
        ollama create sysdesign -f output/gguf/Modelfile 2>/dev/null || true
        python 06_evaluate.py --backend ollama --model sysdesign --limit 10
    fi
else
    echo "Ollama not installed. Skipping auto-evaluation."
    echo "Install Ollama and run: python 06_evaluate.py --backend ollama --model sysdesign"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "═══════════════════════════════════════════"
echo ""
echo "  Artifacts:"
echo "    Training data: data/train.jsonl ($TRAIN_COUNT examples)"
echo "    LoRA adapter:  output/checkpoints/final/"
echo "    Merged model:  output/merged/"
echo "    GGUF model:    output/gguf/"
echo ""
echo "  Next steps:"
echo "    1. Review eval results in output/eval_results/"
echo "    2. Manually score 10+ responses"
echo "    3. If quality is insufficient:"
echo "       - Add more training data (more variants or better prompts)"
echo "       - Try LoRA rank 64 instead of 32"
echo "       - Try 4-5 epochs instead of 3"
echo "    4. Deploy with: ollama run sysdesign"
echo ""
