#!/usr/bin/env python3
"""Generate teacher plans for incomplete commit-pair examples."""
import json
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from repodesign.training.data_gen_commit_pair import generate_commit_pair_example

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

data_dir = Path("data/commit_pairs_production")
pairs_file = data_dir / "commit_pairs.json"

pairs = json.loads(pairs_file.read_text(encoding="utf-8"))
logger.info(f"Loaded {len(pairs)} commit pairs from {pairs_file}")

# Find incomplete examples
incomplete = []
for pair in pairs:
    repo_name = pair["repo_full_name"].split("/")[-1]
    dir_name = f"{repo_name}_pr{pair['pr_number']}"
    example_dir = data_dir / dir_name
    
    if (example_dir / "repo_ir.json").exists() and not (example_dir / "teacher_plan.json").exists():
        incomplete.append((pair, example_dir))

logger.info(f"Found {len(incomplete)} incomplete examples to generate")

# Optional limit for testing
limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(incomplete)
if limit < len(incomplete):
    incomplete = incomplete[:limit]
    logger.info(f"Limiting to first {limit} examples")

# Track stats
success = 0
failed = 0
validated = 0

for i, (pair, example_dir) in enumerate(incomplete):
    logger.info(f"\n[{i+1}/{len(incomplete)}] Generating: {example_dir.name}")
    
    repo_ir = json.loads((example_dir / "repo_ir.json").read_text(encoding="utf-8"))
    
    try:
        result = generate_commit_pair_example(pair, repo_ir, str(example_dir))
        
        if result is None:
            logger.error(f"  ❌ Generation returned None")
            failed += 1
            continue
        
        # Validate immediately
        plan_path = example_dir / "teacher_plan.json"
        spec_path = example_dir / "spec.json"
        diff_path = example_dir / "ground_truth_diff.txt"
        
        if not plan_path.exists() or not spec_path.exists() or not diff_path.exists():
            logger.error(f"  ❌ Missing output files")
            failed += 1
            continue
        
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        manifest = set(repo_ir.get("file_manifest", []))
        
        # Check 1: Has required fields
        if not plan.get("architecture_decisions") or not plan.get("tickets"):
            logger.warning(f"  ⚠️  Missing architecture_decisions or tickets")
        
        # Check 2: files_to_modify all in manifest
        for ticket in plan.get("tickets", []):
            bad_modify = [f for f in ticket.get("files_to_modify", []) if f not in manifest]
            if bad_modify:
                logger.warning(f"  ⚠️  files_to_modify not in manifest: {bad_modify[:2]}")
                break
        else:
            validated += 1
        
        success += 1
        logger.info(f"  ✅ Generated ({len(plan.get('tickets', []))} tickets, {len(plan.get('architecture_decisions', []))} decisions)")
        
    except Exception as e:
        logger.error(f"  ❌ Error: {e}")
        failed += 1
    
    # Progress summary every 10
    if (i + 1) % 10 == 0:
        logger.info(f"\n--- Progress: {i+1}/{len(incomplete)} | ✅ {success} | ❌ {failed} | Validated: {validated} ---\n")

logger.info(f"\n{'='*60}")
logger.info(f"DONE: {success}/{len(incomplete)} generated, {failed} failed, {validated} fully validated")
logger.info(f"{'='*60}")
