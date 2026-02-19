#!/usr/bin/env python3
"""
03_prepare_dataset.py

Converts raw generated conversations + diagram QA pairs into the
training format expected by Unsloth/TRL.

Output: data/train.jsonl (ShareGPT format)

ShareGPT format:
{
    "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."},
        ...
    ]
}
"""

import json
import random
from pathlib import Path

# ─── Config ───
CONVERSATIONS_DIR = Path("data/raw/conversations")
DIAGRAMS_FILE = Path("data/raw/diagram_descriptions.jsonl")
OUTPUT_FILE = Path("data/train.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are an expert systems design assistant. You help engineers design scalable, reliable, and maintainable distributed systems.

When helping with system design, you always:
1. Clarify requirements and estimate scale (back-of-envelope math)
2. Propose a specific architecture with concrete technology choices
3. Explain trade-offs explicitly — why you chose X over Y, and when Y would be better
4. Identify potential bottlenecks, single points of failure, and failure modes
5. Discuss consistency, availability, and partition tolerance trade-offs
6. Provide concrete numbers: QPS, storage requirements, latency targets
7. Adapt your design when requirements change

You are opinionated but always justify your opinions. You prefer proven technologies over trendy ones unless there is a specific reason to use something newer."""


def process_conversations():
    """Load and format conversation files."""
    examples = []
    
    conv_files = list(CONVERSATIONS_DIR.glob("*.json"))
    print(f"Found {len(conv_files)} conversation files")
    
    for fpath in conv_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            
            # Convert to ShareGPT format
            messages = [{"from": "system", "value": SYSTEM_PROMPT}]
            
            for msg in data["conversations"]:
                role = "human" if msg["role"] == "user" else "gpt"
                messages.append({"from": role, "value": msg["content"]})
            
            # Ensure conversation alternates correctly
            valid = True
            for i in range(1, len(messages)):
                expected = "human" if i % 2 == 1 else "gpt"
                if messages[i]["from"] != expected:
                    valid = False
                    break
            
            # Ensure ends with assistant response
            if messages[-1]["from"] != "gpt":
                valid = False
            
            if valid and len(messages) >= 5:  # system + at least 2 exchanges
                examples.append({
                    "id": data.get("id", fpath.stem),
                    "conversations": messages,
                    "source": "conversation",
                    "domain": data.get("domain", "unknown"),
                })
            else:
                print(f"  Skipping {fpath.name}: invalid format")
                
        except Exception as e:
            print(f"  Error processing {fpath.name}: {e}")
    
    return examples


def process_diagram_qa():
    """Convert diagram QA pairs into conversation format."""
    examples = []
    
    if not DIAGRAMS_FILE.exists():
        print(f"No diagram descriptions found at {DIAGRAMS_FILE}")
        return examples
    
    with open(DIAGRAMS_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                
                # Create a conversation from each QA pair
                for qa in data.get("qa_pairs", []):
                    messages = [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {
                            "from": "human",
                            "value": f"[Looking at an architecture diagram]\n\n{qa['q']}\n\nContext about the diagram:\n- Architecture type: {data['architecture_type']}\n- Components: {json.dumps(data['components'], indent=2)}\n- Data flow: {data['data_flow']}"
                        },
                        {
                            "from": "gpt",
                            "value": qa["a"]
                        }
                    ]
                    
                    examples.append({
                        "id": f"{data['id']}_qa",
                        "conversations": messages,
                        "source": "diagram_qa",
                        "domain": data.get("architecture_type", "unknown"),
                    })
                
                # Create an analysis conversation from the full description
                components_text = "\n".join(f"- {k}: {v}" for k, v in data["components"].items())
                bottlenecks_text = "\n".join(f"- {b}" for b in data.get("potential_bottlenecks", []))
                improvements_text = "\n".join(f"- {i}" for i in data.get("improvements", []))
                
                messages = [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {
                        "from": "human",
                        "value": f"I have this architecture and I want your analysis. Here are the components:\n\n{components_text}\n\nData flow: {data['data_flow']}\n\nWhat are the weaknesses and how would you improve it?"
                    },
                    {
                        "from": "gpt",
                        "value": f"Here is my analysis of this {data['architecture_type']} architecture:\n\n**Potential Bottlenecks:**\n{bottlenecks_text}\n\n**Single Points of Failure:**\n" + "\n".join(f"- {s}" for s in data.get("single_points_of_failure", [])) + f"\n\n**Recommended Improvements:**\n{improvements_text}\n\nThe priority depends on your current traffic and growth rate. If you are seeing latency issues now, start with caching. If you are worried about availability, focus on eliminating single points of failure first."
                    }
                ]
                
                examples.append({
                    "id": f"{data['id']}_analysis",
                    "conversations": messages,
                    "source": "diagram_analysis",
                    "domain": data.get("architecture_type", "unknown"),
                })
                
            except Exception as e:
                print(f"  Error processing diagram: {e}")
    
    return examples


def add_checklist_conversations():
    """Add conversations that teach the model about the systems design checklist."""
    
    checklist_examples = [
        {
            "human": "I need to design a new system. Where do I even start?",
            "gpt": """Start with these five things before touching any technology:

**1. Problem statement in one sentence.** If you cannot state what you are building in one sentence, the scope is too vague. Example: "A system that lets 10M users share and view short videos with a personalized feed."

**2. Core use cases (3-5 max).** What are the critical operations? For a video platform: upload video, view feed, search videos, follow users. Everything else is secondary.

**3. Scale estimates.** How many users? DAU? Requests per second? Data size? Growth rate? Do the back-of-envelope math now. This eliminates bad choices early.

**4. Non-functional requirements.** What is your latency target (p99)? What uptime do you need (99.9% vs 99.99%)? Do you need strong consistency or is eventual OK? These shape every architectural decision.

**5. Constraints.** Budget, team size, existing infrastructure, regulatory requirements. A 3-person team should not build a microservices architecture with 20 services.

Only after you have clear answers to these five should you start choosing technologies and drawing architecture diagrams. Want to walk through these for your specific system?"""
        },
        {
            "human": "How do I decide between SQL and NoSQL for my project?",
            "gpt": """This is one of the most common decisions, and the answer depends on your access patterns, not your data complexity. Here is the framework I use:

**Choose a relational database (PostgreSQL, MySQL) when:**
- You need complex queries with joins across multiple entities
- You need strong ACID transactions (financial data, inventory management)
- Your schema is well-defined and relationships are important
- Your read/write ratio is balanced or read-heavy at moderate scale
- You are not sure what you need (relational is the safer default)

**Choose a document store (MongoDB, DynamoDB) when:**
- Your data has a natural tree/document structure (user profiles with nested preferences)
- You typically read/write entire documents, not individual fields
- You need flexible schemas that evolve frequently
- You need horizontal scaling for massive write throughput
- Your queries are primarily key-based lookups

**Choose a wide-column store (Cassandra, ScyllaDB) when:**
- You have massive write throughput (100K+ writes/sec)
- You can tolerate eventual consistency
- Your queries are predictable (you know your access patterns upfront)
- Time-series data or event logging

**The trade-off that matters most:** SQL databases give you flexibility in queries but are harder to scale horizontally. NoSQL databases scale easily but restrict how you can query the data. If you choose NoSQL, you are committing to specific access patterns. Changing those patterns later is painful.

My default recommendation: start with PostgreSQL unless you have a specific reason not to. It handles more use cases than people think, scales further than people expect (easily to millions of rows, often billions with partitioning), and gives you the most flexibility. You can always add a specialized store later for specific workloads."""
        },
        {
            "human": "When should I add caching to my system?",
            "gpt": """Add caching when you have a specific performance problem, not as a default architecture choice. Caching adds complexity (invalidation, consistency, cold starts), so it should solve a real problem.

**Add caching when:**
- Your database is becoming a bottleneck (high CPU, slow queries)
- You have hot data that is read much more than it is written (100:1 read/write ratio or higher)
- The same expensive computation is repeated for many users (e.g., a trending posts list)
- Your latency targets require sub-millisecond reads that a database cannot provide

**Do NOT add caching when:**
- You have not measured whether the database is actually the bottleneck (it might be your application code)
- Your data changes frequently and must always be fresh (real-time stock prices, live scores)
- Your access pattern is mostly unique lookups with low repeat rate (cache hit rate would be too low)
- You have not optimized your database queries first (add indexes before adding cache)

**Cache invalidation strategy matters more than the cache itself:**
- Cache-aside with TTL: simplest, good enough for most cases. Set TTL to the staleness you can tolerate.
- Write-through: write to cache and DB together. Always fresh but adds write latency.
- Event-driven invalidation: when data changes, publish event, invalidate cache. Best consistency but most complex.

**Practical numbers:** Redis on a single instance can handle 100K+ operations/second. If your total read QPS is under 10K, you probably do not need Redis — optimize your database first. If you are at 50K+ reads/second with hot keys, caching becomes essential.

Where to cache: client-side (browser) > CDN (static/semi-static) > application cache (Redis) > database query cache. Start from the outermost layer."""
        },
    ]
    
    examples = []
    for i, ex in enumerate(checklist_examples):
        examples.append({
            "id": f"checklist_{i:03d}",
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": ex["human"]},
                {"from": "gpt", "value": ex["gpt"]},
            ],
            "source": "checklist",
            "domain": "general",
        })
    
    return examples


def main():
    all_examples = []
    
    # 1. Process generated conversations
    print("Processing conversations...")
    convs = process_conversations()
    print(f"  {len(convs)} conversations")
    all_examples.extend(convs)
    
    # 2. Process diagram QA
    print("Processing diagram QA pairs...")
    diagrams = process_diagram_qa()
    print(f"  {len(diagrams)} diagram examples")
    all_examples.extend(diagrams)
    
    # 3. Add checklist knowledge
    print("Adding checklist conversations...")
    checklist = add_checklist_conversations()
    print(f"  {len(checklist)} checklist examples")
    all_examples.extend(checklist)
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for ex in all_examples:
            # Write only the conversations field (what the trainer needs)
            f.write(json.dumps({"conversations": ex["conversations"]}) + "\n")
    
    print(f"\nTotal training examples: {len(all_examples)}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Stats
    sources = {}
    domains = {}
    for ex in all_examples:
        src = ex.get("source", "unknown")
        dom = ex.get("domain", "unknown")
        sources[src] = sources.get(src, 0) + 1
        domains[dom] = domains.get(dom, 0) + 1
    
    print(f"\nBy source:")
    for k, v in sorted(sources.items()):
        print(f"  {k}: {v}")
    
    print(f"\nBy domain:")
    for k, v in sorted(domains.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
