#!/usr/bin/env python3
"""
05_merge_and_export.py

Merges the LoRA adapter with the base model and exports to:
1. Full merged model (for vLLM serving)
2. GGUF quantized model (for Ollama / llama.cpp deployment)

Run this on the same GPU machine after training.

Usage:
    python 05_merge_and_export.py                    # Default
    python 05_merge_and_export.py --quant q4_k_m     # Specific quantization
    python 05_merge_and_export.py --skip-merged       # Only export GGUF
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge adapter and export model")
    parser.add_argument("--adapter", default="output/checkpoints/final",
                        help="Path to the saved LoRA adapter")
    parser.add_argument("--base-model", default="unsloth/Qwen2-VL-7B-Instruct",
                        help="Base model name")
    parser.add_argument("--merged-output", default="output/merged",
                        help="Output path for merged model")
    parser.add_argument("--gguf-output", default="output/gguf",
                        help="Output path for GGUF files")
    parser.add_argument("--quant", default="q4_k_m",
                        choices=["q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="GGUF quantization level")
    parser.add_argument("--skip-merged", action="store_true",
                        help="Skip saving full merged model (saves disk)")
    args = parser.parse_args()

    from unsloth import FastVisionModel

    # ─── Step 1: Load base model + adapter ───
    print("=" * 60)
    print("STEP 1: Loading base model and adapter")
    print("=" * 60)

    model, tokenizer = FastVisionModel.from_pretrained(
        args.adapter,            # This loads base + adapter
        load_in_4bit=True,
    )

    # ─── Step 2: Save merged model (optional) ───
    if not args.skip_merged:
        print("=" * 60)
        print("STEP 2: Saving merged model (16-bit)")
        print("=" * 60)

        Path(args.merged_output).mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(
            args.merged_output,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"Merged model saved to {args.merged_output}")
    else:
        print("Skipping merged model save")

    # ─── Step 3: Export GGUF ───
    print("=" * 60)
    print(f"STEP 3: Exporting GGUF ({args.quant})")
    print("=" * 60)

    Path(args.gguf_output).mkdir(parents=True, exist_ok=True)

    # Note: Unsloth handles GGUF conversion internally
    # For Qwen2-VL, you may need to use llama.cpp directly
    # This is because vision models need special handling for GGUF
    
    try:
        model.save_pretrained_gguf(
            args.gguf_output,
            tokenizer,
            quantization_method=args.quant,
        )
        print(f"GGUF saved to {args.gguf_output}")
    except Exception as e:
        print(f"Unsloth GGUF export failed: {e}")
        print("\nFalling back to manual conversion...")
        print("You can convert manually with llama.cpp:")
        print(f"  1. git clone https://github.com/ggerganov/llama.cpp")
        print(f"  2. cd llama.cpp && make")
        print(f"  3. python convert_hf_to_gguf.py {args.merged_output}")
        print(f"  4. ./llama-quantize model.gguf model-{args.quant}.gguf {args.quant}")

    # ─── Step 4: Create Ollama Modelfile ───
    print("=" * 60)
    print("STEP 4: Creating Ollama Modelfile")
    print("=" * 60)

    # Find the GGUF file
    gguf_files = list(Path(args.gguf_output).glob("*.gguf"))
    gguf_path = gguf_files[0] if gguf_files else f"{args.gguf_output}/model-{args.quant}.gguf"

    modelfile_content = f"""FROM {gguf_path}

TEMPLATE \"\"\"{{{{- if .System }}}}
<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"You are an expert systems design assistant. You help engineers design scalable, reliable, and maintainable distributed systems. You always discuss trade-offs, give specific technology recommendations with justification, include concrete numbers, and identify failure modes proactively.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
"""

    modelfile_path = Path(args.gguf_output) / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"Modelfile saved to {modelfile_path}")
    print()
    print("To deploy with Ollama:")
    print(f"  1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
    print(f"  2. ollama create sysdesign -f {modelfile_path}")
    print(f"  3. ollama run sysdesign")
    print()
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
