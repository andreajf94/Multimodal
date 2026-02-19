#!/usr/bin/env python3
"""
02_generate_conversations.py

Uses Claude API (or OpenAI) to generate high-quality multi-turn
system design conversations from the seed prompts.

Cost estimate: ~$50-100 for 1000-2000 conversations using Claude Sonnet.

Usage:
    export ANTHROPIC_API_KEY=your_key_here
    python 02_generate_conversations.py --provider anthropic --count 20

    # Or with OpenAI:
    export OPENAI_API_KEY=your_key_here
    python 02_generate_conversations.py --provider openai --count 20
"""

import json
import os
import sys
import time
import argparse
import random
from pathlib import Path

# ─── Config ───
OUTPUT_DIR = Path("data/raw/conversations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_FILE = Path("prompts/system_design_prompts.json")

# The meta-prompt that instructs the AI to generate training conversations
META_PROMPT = """You are generating training data for a systems design AI assistant.

Given the system design prompt below, generate a realistic multi-turn conversation
between a User and an expert Systems Design Assistant. The conversation should:

1. Start with the user asking the design question
2. The assistant should first clarify scale and requirements, do back-of-envelope estimates
3. Propose a high-level architecture with specific technology choices
4. Discuss trade-offs explicitly (why X over Y, under what conditions Y would be better)
5. The user should ask follow-up questions pushing on specific areas
6. The assistant should deep dive into at least 3 of these: data model, caching, partitioning,
   replication, failure modes, consistency, API design, or message queues
7. The user should change or add a requirement mid-conversation
8. The assistant should adapt the design and explain what changes

Requirements for the assistant's responses:
- Give SPECIFIC technology recommendations (not "use a NoSQL database" but "use DynamoDB because...")
- Include concrete numbers (QPS, storage, latency targets)
- Always discuss trade-offs and alternatives
- Mention failure modes proactively
- Be opinionated but justify opinions
- When discussing scale, show the math

The conversation should be 8-14 turns total (4-7 exchanges).

Output format: JSON array of message objects with "role" (user/assistant) and "content" fields.

SYSTEM DESIGN PROMPT:
{prompt}

KEY TOPICS TO COVER: {topics}

Generate the conversation now as a JSON array:"""


def call_anthropic(prompt: str, max_retries: int = 3) -> str:
    """Call Claude API to generate a conversation."""
    try:
        import anthropic
    except ImportError:
        print("pip install anthropic")
        sys.exit(1)
    
    client = anthropic.Anthropic()
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                print(f"  Retry in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def call_openai(prompt: str, max_retries: int = 3) -> str:
    """Call OpenAI API to generate a conversation."""
    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai")
        sys.exit(1)
    
    client = OpenAI()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 10
                print(f"  Retry in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def parse_conversation(raw_text: str) -> list:
    """Extract JSON conversation from the API response."""
    # Try to find JSON array in the response
    text = raw_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]") + 1
    
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")
    
    json_str = text[start:end]
    conversation = json.loads(json_str)
    
    # Validate structure
    for msg in conversation:
        assert "role" in msg, "Missing 'role' field"
        assert "content" in msg, "Missing 'content' field"
        assert msg["role"] in ("user", "assistant"), f"Invalid role: {msg['role']}"
    
    return conversation


def generate_conversations(provider: str, count_per_prompt: int, max_prompts: int):
    """Main generation loop."""
    
    # Load prompts
    with open(PROMPTS_FILE) as f:
        prompts_data = json.load(f)
    
    prompts = prompts_data["system_design_prompts"]
    if max_prompts:
        prompts = prompts[:max_prompts]
    
    call_fn = call_anthropic if provider == "anthropic" else call_openai
    
    total_generated = 0
    total_failed = 0
    
    for prompt_data in prompts:
        prompt_id = prompt_data["id"]
        prompt_text = prompt_data["prompt"]
        topics = ", ".join(prompt_data["key_topics"])
        
        for variant in range(count_per_prompt):
            output_file = OUTPUT_DIR / f"{prompt_id}_v{variant:02d}.json"
            
            # Skip if already generated
            if output_file.exists():
                print(f"  Skip {output_file.name} (exists)")
                total_generated += 1
                continue
            
            print(f"  Generating {prompt_id} variant {variant}...")
            
            try:
                meta = META_PROMPT.format(prompt=prompt_text, topics=topics)
                raw = call_fn(meta)
                conversation = parse_conversation(raw)
                
                # Save
                result = {
                    "id": f"{prompt_id}_v{variant:02d}",
                    "source_prompt_id": prompt_id,
                    "domain": prompt_data["domain"],
                    "difficulty": prompt_data["difficulty"],
                    "key_topics": prompt_data["key_topics"],
                    "conversations": conversation,
                    "num_turns": len(conversation),
                }
                
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                
                total_generated += 1
                print(f"    OK ({len(conversation)} turns)")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                total_failed += 1
                print(f"    FAILED: {e}")
    
    print(f"\nDone. Generated: {total_generated}, Failed: {total_failed}")
    print(f"Output directory: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Generate training conversations")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--count", type=int, default=20,
                        help="Variants per prompt (total = count * num_prompts)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit number of seed prompts (default: all 50)")
    args = parser.parse_args()
    
    # Check API key
    if args.provider == "anthropic":
        assert os.environ.get("ANTHROPIC_API_KEY"), "Set ANTHROPIC_API_KEY"
    else:
        assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY"
    
    print(f"Provider: {args.provider}")
    print(f"Variants per prompt: {args.count}")
    
    with open(PROMPTS_FILE) as f:
        num_prompts = len(json.load(f)["system_design_prompts"])
    if args.max_prompts:
        num_prompts = min(num_prompts, args.max_prompts)
    
    total = num_prompts * args.count
    print(f"Total conversations to generate: {total}")
    
    # Cost estimate
    # ~2000 input tokens + ~3000 output tokens per conversation
    # Claude Sonnet: $3/M input, $15/M output
    # GPT-4o: $2.5/M input, $10/M output
    if args.provider == "anthropic":
        cost = total * (2000 * 3 / 1_000_000 + 3000 * 15 / 1_000_000)
    else:
        cost = total * (2000 * 2.5 / 1_000_000 + 3000 * 10 / 1_000_000)
    
    print(f"Estimated API cost: ${cost:.2f}")
    print()
    
    input("Press Enter to start (or Ctrl+C to cancel)...")
    print()
    
    generate_conversations(args.provider, args.count, args.max_prompts)


if __name__ == "__main__":
    main()
