"""Batch generate file_candidates.json for all commit pair repos."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.repodesign.training.reward import parse_diff_files

def get_file_directory(filepath: str) -> str:
    """Get the directory path from a file path."""
    return str(Path(filepath).parent)

def get_similar_files(target_file: str, all_files: list[str], n: int = 3) -> list[str]:
    """Find files similar to target (same directory or similar extension)."""
    target_dir = get_file_directory(target_file)
    target_ext = Path(target_file).suffix
    
    candidates = []
    
    # First: files in same directory
    same_dir = [f for f in all_files if get_file_directory(f) == target_dir and f != target_file]
    candidates.extend(same_dir[:n])
    
    # If not enough, add files with same extension
    if len(candidates) < n:
        same_ext = [f for f in all_files 
                   if Path(f).suffix == target_ext 
                   and f != target_file 
                   and f not in candidates]
        candidates.extend(same_ext[:n - len(candidates)])
    
    return candidates[:n]

def generate_candidates(
    diff_files_modified: list[str],
    diff_files_created: list[str],
    file_manifest: list[str],
    max_candidates: int = 15,
) -> list[str]:
    """Generate a candidate list: actual files + similar distractors."""
    actual_files = diff_files_modified + diff_files_created
    candidates = set(actual_files)
    
    # Calculate how many distractors to add
    n_actual = len(actual_files)
    n_distractors = max_candidates - n_actual
    
    if n_distractors <= 0:
        return list(candidates)[:max_candidates]
    
    # Add similar files as distractors (3-4 per actual file)
    distractors_per_file = max(1, n_distractors // max(1, n_actual))
    
    for actual_file in actual_files:
        similar = get_similar_files(actual_file, file_manifest, distractors_per_file)
        candidates.update(similar)
        
        if len(candidates) >= max_candidates:
            break
    
    # If still not enough, add random files from repo
    if len(candidates) < max_candidates:
        remaining = max_candidates - len(candidates)
        available = [f for f in file_manifest if f not in candidates]
        if available:
            random_extras = random.sample(available, min(remaining, len(available)))
            candidates.update(random_extras)
    
    return sorted(candidates)[:max_candidates]


def process_repo(repo_dir: Path, max_candidates: int = 15) -> dict:
    """Process a single repo and generate candidate files."""
    # Load diff
    with open(repo_dir / "ground_truth_diff.txt", encoding="utf-8") as f:
        diff_text = f.read()
    
    # Load repo IR
    with open(repo_dir / "repo_ir.json", encoding="utf-8") as f:
        repo_ir = json.load(f)
    
    # Parse diff files
    diff_files = parse_diff_files(diff_text)
    
    # Generate candidates
    candidates = generate_candidates(
        diff_files["modified"],
        diff_files["created"],
        repo_ir["file_manifest"],
        max_candidates=max_candidates,
    )
    
    return {
        "candidates": candidates,
        "n_actual": len(diff_files["modified"]) + len(diff_files["created"]),
        "n_total": len(candidates)
    }


if __name__ == "__main__":
    data_dir = Path("data/commit_pairs_production")
    repos = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"Processing {len(repos)} repos...")
    
    for repo_dir in repos:
        try:
            result = process_repo(repo_dir, max_candidates=15)
            
            # Save to file
            output_path = repo_dir / "file_candidates.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result["candidates"], f, indent=2)
            
            print(f"✓ {repo_dir.name:30s} {result['n_actual']:2d} actual → {result['n_total']:2d} candidates")
            
        except Exception as e:
            print(f"✗ {repo_dir.name:30s} ERROR: {e}")
    
    print("\nDone!")
