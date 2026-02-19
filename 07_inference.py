#!/usr/bin/env python3
"""
07_inference.py

Interactive chat with your fine-tuned systems design model.
Supports Ollama (recommended) and direct transformers loading.

Usage:
    # With Ollama (recommended - fast, easy)
    python 07_inference.py --backend ollama --model sysdesign

    # With transformers (needs GPU)
    python 07_inference.py --backend transformers --model output/merged

    # Single question mode
    python 07_inference.py --backend ollama --model sysdesign \
        --question "Design a URL shortener for 100M URLs"
"""

import argparse
import subprocess
import sys


SYSTEM_PROMPT = """You are an expert systems design assistant. You help engineers design scalable, reliable, and maintainable distributed systems. You always discuss trade-offs, give specific technology recommendations with justification, include concrete numbers, and identify failure modes proactively."""


class OllamaBackend:
    def __init__(self, model_name):
        self.model_name = model_name
        self.history = []
    
    def chat(self, user_message: str) -> str:
        # Build conversation with history
        prompt_parts = [f"System: {SYSTEM_PROMPT}\n"]
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {msg['content']}\n")
        prompt_parts.append(f"User: {user_message}\nAssistant:")
        
        full_prompt = "\n".join(prompt_parts)
        
        result = subprocess.run(
            ["ollama", "run", self.model_name, full_prompt],
            capture_output=True, text=True, timeout=180,
        )
        
        response = result.stdout.strip()
        
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def reset(self):
        self.history = []


class TransformersBackend:
    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Loading model (this may take a minute)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.history = []
        print("Model loaded!")
    
    def chat(self, user_message: str) -> str:
        import torch
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def reset(self):
        self.history = []


def interactive_chat(backend):
    """Run interactive multi-turn chat."""
    print("=" * 60)
    print("SYSTEMS DESIGN ASSISTANT")
    print("=" * 60)
    print("Ask me to design any system. I'll help with architecture,")
    print("trade-offs, scaling, and implementation details.")
    print()
    print("Commands:")
    print("  /reset  - Start a new conversation")
    print("  /quit   - Exit")
    print("=" * 60)
    print()
    
    while True:
        try:
            user_input = input("\033[1;36mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "/quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "/reset":
            backend.reset()
            print("\n--- Conversation reset ---\n")
            continue
        
        print()
        try:
            response = backend.chat(user_input)
            print(f"\033[1;32mAssistant:\033[0m {response}")
        except Exception as e:
            print(f"\033[1;31mError:\033[0m {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Chat with your systems design model")
    parser.add_argument("--backend", choices=["ollama", "transformers"], default="ollama")
    parser.add_argument("--model", default="sysdesign",
                        help="Model name (Ollama) or path (transformers)")
    parser.add_argument("--question", default=None,
                        help="Single question mode (no interactive chat)")
    args = parser.parse_args()
    
    # Initialize backend
    if args.backend == "ollama":
        backend = OllamaBackend(args.model)
    else:
        backend = TransformersBackend(args.model)
    
    # Single question mode
    if args.question:
        response = backend.chat(args.question)
        print(response)
        return
    
    # Interactive mode
    interactive_chat(backend)


if __name__ == "__main__":
    main()
