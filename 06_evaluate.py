#!/usr/bin/env python3
"""
06_evaluate.py

Runs the fine-tuned model on held-out evaluation prompts
and saves results for human review.

Can run against:
  - The fine-tuned model locally (via Ollama or transformers)
  - A baseline model for comparison

Usage:
    python 06_evaluate.py --backend ollama --model sysdesign
    python 06_evaluate.py --backend transformers --model output/merged
    python 06_evaluate.py --backend ollama --model sysdesign --compare qwen2.5:7b
"""

import json
import time
import argparse
import csv
from pathlib import Path
from datetime import datetime

EVAL_PROMPTS_FILE = Path("prompts/eval_prompts.json")
RESULTS_DIR = Path("output/eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are an expert systems design assistant. You help engineers design scalable, reliable, and maintainable distributed systems. You always discuss trade-offs, give specific technology recommendations with justification, include concrete numbers, and identify failure modes proactively."""


def call_ollama(prompt: str, model: str) -> str:
    """Call a local Ollama model."""
    import subprocess
    
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True, text=True, timeout=120
    )
    return result.stdout.strip()


def call_transformers(prompt: str, model_path: str, tokenizer=None, model=None):
    """Call the merged model directly with transformers."""
    if model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def evaluate_response(prompt_data: dict, response: str) -> dict:
    """Auto-score a response based on simple heuristics.
    
    These are rough proxies — human evaluation is still needed.
    """
    scores = {}
    text_lower = response.lower()
    
    # 1. Specificity: Does it mention specific technologies?
    technologies = [
        "postgresql", "mysql", "mongodb", "dynamodb", "cassandra", "redis",
        "elasticsearch", "kafka", "rabbitmq", "sqs", "kinesis",
        "kubernetes", "docker", "terraform", "nginx", "haproxy",
        "s3", "cloudfront", "lambda", "ec2", "rds",
        "grpc", "graphql", "rest", "websocket",
    ]
    tech_count = sum(1 for t in technologies if t in text_lower)
    scores["specificity"] = min(5, tech_count)  # 0-5
    
    # 2. Trade-off awareness: Does it discuss alternatives?
    tradeoff_phrases = [
        "trade-off", "tradeoff", "trade off",
        "alternatively", "on the other hand", "the downside",
        "however", "the cost is", "the risk is",
        "if instead", "compared to", "rather than",
        "pros and cons", "advantage", "disadvantage",
    ]
    tradeoff_count = sum(1 for t in tradeoff_phrases if t in text_lower)
    scores["tradeoff_awareness"] = min(5, tradeoff_count)
    
    # 3. Quantitative reasoning: Does it include numbers?
    import re
    numbers = re.findall(r'\d+[KMBkmb]?\s*(?:req|qps|rps|ops|users|DAU|MAU|writes|reads|bytes|GB|TB|PB|MB|ms|sec|min|%)', text_lower)
    scores["quantitative"] = min(5, len(numbers))
    
    # 4. Failure mode awareness
    failure_phrases = [
        "single point of failure", "spof", "if.*goes down", "if.*fails",
        "failure mode", "failover", "redundancy", "circuit breaker",
        "retry", "timeout", "cascading", "bottleneck",
    ]
    failure_count = sum(1 for f in failure_phrases if re.search(f, text_lower))
    scores["failure_awareness"] = min(5, failure_count)
    
    # 5. Topic coverage: Does it address expected topics?
    expected = prompt_data.get("expected_topics", [])
    covered = sum(1 for topic in expected if topic.lower().replace("_", " ") in text_lower or topic.lower() in text_lower)
    scores["topic_coverage"] = round(5 * covered / max(len(expected), 1), 1)
    
    # 6. Length (proxy for completeness)
    word_count = len(response.split())
    if word_count < 100:
        scores["completeness"] = 1
    elif word_count < 300:
        scores["completeness"] = 3
    elif word_count < 800:
        scores["completeness"] = 5
    else:
        scores["completeness"] = 4  # Too long might be rambling
    
    # Overall (simple average)
    scores["overall"] = round(sum(scores.values()) / len(scores), 1)
    
    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--backend", choices=["ollama", "transformers"], default="ollama")
    parser.add_argument("--model", default="sysdesign",
                        help="Model name (Ollama) or path (transformers)")
    parser.add_argument("--compare", default=None,
                        help="Baseline model to compare against")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of eval prompts")
    args = parser.parse_args()
    
    # Load eval prompts
    with open(EVAL_PROMPTS_FILE) as f:
        eval_data = json.load(f)
    
    prompts = eval_data["eval_prompts"]
    if args.limit:
        prompts = prompts[:args.limit]
    
    print(f"Evaluating {len(prompts)} prompts with {args.backend} ({args.model})")
    
    # Set up call function
    if args.backend == "ollama":
        call_fn = lambda p: call_ollama(p, args.model)
    else:
        # Load model once
        call_fn = lambda p: call_transformers(p, args.model)
    
    # Run evaluation
    results = []
    
    for i, prompt_data in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt_data['id']}")
        print(f"  Prompt: {prompt_data['prompt'][:80]}...")
        
        start = time.time()
        try:
            response = call_fn(prompt_data["prompt"])
            elapsed = time.time() - start
            
            scores = evaluate_response(prompt_data, response)
            
            result = {
                "id": prompt_data["id"],
                "prompt": prompt_data["prompt"],
                "response": response,
                "scores": scores,
                "elapsed_seconds": round(elapsed, 1),
                "word_count": len(response.split()),
                "model": args.model,
            }
            results.append(result)
            
            print(f"  Score: {scores['overall']}/5 | Words: {result['word_count']} | Time: {elapsed:.1f}s")
            print(f"  Breakdown: spec={scores['specificity']} tradeoff={scores['tradeoff_awareness']} "
                  f"quant={scores['quantitative']} failure={scores['failure_awareness']} "
                  f"topics={scores['topic_coverage']} complete={scores['completeness']}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": prompt_data["id"],
                "prompt": prompt_data["prompt"],
                "response": f"ERROR: {e}",
                "scores": {"overall": 0},
                "model": args.model,
            })
    
    # ─── Compare with baseline if requested ───
    if args.compare:
        print(f"\n\nRunning baseline comparison with {args.compare}...")
        baseline_fn = lambda p: call_ollama(p, args.compare)
        
        baseline_results = []
        for i, prompt_data in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] {prompt_data['id']}")
            try:
                response = baseline_fn(prompt_data["prompt"])
                scores = evaluate_response(prompt_data, response)
                baseline_results.append(scores)
            except Exception as e:
                print(f"    ERROR: {e}")
                baseline_results.append({"overall": 0})
    
    # ─── Save results ───
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full results as JSON
    results_file = RESULTS_DIR / f"eval_{args.model.replace('/', '_')}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: {results_file}")
    
    # Summary CSV for spreadsheet review
    csv_file = RESULTS_DIR / f"eval_{args.model.replace('/', '_')}_{timestamp}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Overall", "Specificity", "TradeOffs", "Quantitative",
                         "FailureAwareness", "TopicCoverage", "Completeness", "Words", "Time(s)"])
        for r in results:
            s = r["scores"]
            writer.writerow([
                r["id"], s.get("overall", 0), s.get("specificity", 0),
                s.get("tradeoff_awareness", 0), s.get("quantitative", 0),
                s.get("failure_awareness", 0), s.get("topic_coverage", 0),
                s.get("completeness", 0), r.get("word_count", 0),
                r.get("elapsed_seconds", 0),
            ])
    print(f"Summary CSV: {csv_file}")
    
    # ─── Print summary ───
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    valid = [r for r in results if r["scores"].get("overall", 0) > 0]
    if valid:
        avg_overall = sum(r["scores"]["overall"] for r in valid) / len(valid)
        avg_words = sum(r.get("word_count", 0) for r in valid) / len(valid)
        
        print(f"  Model: {args.model}")
        print(f"  Prompts evaluated: {len(valid)}/{len(prompts)}")
        print(f"  Average overall score: {avg_overall:.1f}/5")
        print(f"  Average response length: {avg_words:.0f} words")
        
        # Score distribution
        for dim in ["specificity", "tradeoff_awareness", "quantitative",
                     "failure_awareness", "topic_coverage", "completeness"]:
            avg = sum(r["scores"].get(dim, 0) for r in valid) / len(valid)
            print(f"  {dim:>20}: {avg:.1f}/5")
    
    if args.compare and baseline_results:
        print(f"\n  Baseline ({args.compare}):")
        baseline_valid = [b for b in baseline_results if b.get("overall", 0) > 0]
        if baseline_valid:
            baseline_avg = sum(b["overall"] for b in baseline_valid) / len(baseline_valid)
            print(f"  Average overall: {baseline_avg:.1f}/5")
            print(f"  Delta: {avg_overall - baseline_avg:+.1f}")

    print("\n  NOTE: Auto-scores are rough proxies. Review the full JSON output")
    print("  and manually score 10+ responses for reliable quality assessment.")


if __name__ == "__main__":
    main()
