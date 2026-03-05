#!/usr/bin/env python3
"""Inspect and display scraped repo data quality before full-scale scraping.

Usage:
    python scripts/inspect_data.py [data/repo_irs]
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

# ─── Colors for terminal output ───
class C:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    END = "\033[0m"


def load_json(path: Path) -> dict | list | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    else:
        return f"{n / (1024 * 1024):.1f} MB"


def check_repo_ir(ir: dict) -> list[str]:
    """Return list of quality warnings for a RepoIR."""
    warnings = []
    meta = ir.get("repo_metadata", {})

    if not meta.get("name"):
        warnings.append("Missing repo name")
    if meta.get("num_contributors", 0) == 0:
        warnings.append("num_contributors=0 (enrichment may have failed)")
    if meta.get("total_loc", 0) == 0:
        warnings.append("total_loc=0 (no code found)")
    if not ir.get("dependencies"):
        warnings.append("No dependencies extracted")
    if not ir.get("api_routes"):
        warnings.append("No API routes found")
    if not ir.get("data_models"):
        warnings.append("No data models found")
    if ir.get("architectural_summary", "").startswith("(LLM summary skipped"):
        warnings.append("LLM architectural summary skipped")
    if not ir.get("file_manifest"):
        warnings.append("Empty file manifest")
    if not ir.get("diagram_paths"):
        warnings.append("No diagrams")

    infra = ir.get("infrastructure", {})
    if not infra.get("ci_cd"):
        warnings.append("No CI/CD detected")

    return warnings


def check_spec(spec: dict) -> list[str]:
    warnings = []
    if not spec.get("project_name"):
        warnings.append("Missing project_name")
    if not spec.get("feature_name"):
        warnings.append("Missing feature_name")
    if not spec.get("description"):
        warnings.append("Missing description")
    reqs = spec.get("functional_requirements", [])
    if len(reqs) < 3:
        warnings.append(f"Only {len(reqs)} functional requirements (want >=3)")
    if not spec.get("scale_tier"):
        warnings.append("Missing scale_tier")
    return warnings


def check_teacher_plan(plan: dict) -> list[str]:
    warnings = []
    decisions = plan.get("architecture_decisions", [])
    tickets = plan.get("tickets", [])

    if len(decisions) < 2:
        warnings.append(f"Only {len(decisions)} architecture decisions (want >=3)")
    if len(tickets) < 3:
        warnings.append(f"Only {len(tickets)} tickets (want >=4)")
    if not plan.get("technology_choices"):
        warnings.append("No technology choices")

    # Check file path grounding
    all_paths = set()
    for d in decisions:
        all_paths.update(d.get("files_affected", []))
    for t in tickets:
        all_paths.update(t.get("files_to_modify", []))
        all_paths.update(t.get("files_to_create", []))

    if len(all_paths) < 3:
        warnings.append(f"Only {len(all_paths)} unique file paths referenced")

    return warnings


def print_section(title: str):
    print(f"\n{C.BOLD}{C.CYAN}{'═' * 60}{C.END}")
    print(f"{C.BOLD}{C.CYAN}  {title}{C.END}")
    print(f"{C.BOLD}{C.CYAN}{'═' * 60}{C.END}")


def print_warnings(warnings: list[str], label: str):
    if not warnings:
        print(f"  {C.GREEN}✓ {label}: All checks passed{C.END}")
    else:
        print(f"  {C.YELLOW}⚠ {label}:{C.END}")
        for w in warnings:
            print(f"    {C.DIM}• {w}{C.END}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inspect scraped repo data quality")
    parser.add_argument("repo_irs_dir", nargs="?", default="data/repo_irs", help="Directory with extracted repo IRs")
    parser.add_argument("--repo-list", default=None, help="Path to repo_list.json (auto-detected if not given)")
    args = parser.parse_args()

    base = Path(args.repo_irs_dir)
    # Auto-detect repo_list path: check for repo_list_test.json next to repo_irs_test, etc.
    if args.repo_list:
        repo_list_path = Path(args.repo_list)
    else:
        # Try matching name: data/repo_irs_test -> data/repo_list_test.json
        dir_name = base.name  # e.g. "repo_irs_test" or "repo_irs"
        suffix = dir_name.replace("repo_irs", "")  # e.g. "_test" or ""
        candidate = base.parent / f"repo_list{suffix}.json"
        repo_list_path = candidate if candidate.exists() else Path("data/repo_list.json")

    # ─── Repo List Overview ───
    print_section("REPO LIST OVERVIEW")
    repo_list = load_json(repo_list_path)
    if repo_list:
        print(f"  Source: {repo_list_path}")
        print(f"  Total repos scraped: {C.BOLD}{len(repo_list)}{C.END}")

        # Aggregate stats
        langs = Counter(r.get("primary_language", "unknown") for r in repo_list)
        tiers = Counter(r.get("scale_tier", "unknown") for r in repo_list)
        stars = [r.get("star_count", 0) for r in repo_list]
        sizes = [r.get("size_kb", 0) for r in repo_list]

        print(f"\n  {C.BOLD}Languages:{C.END}")
        for lang, count in langs.most_common():
            print(f"    {lang}: {count}")

        print(f"\n  {C.BOLD}Scale Tiers:{C.END}")
        for tier, count in tiers.most_common():
            print(f"    {tier}: {count}")

        print(f"\n  {C.BOLD}Stars:{C.END}")
        print(f"    Min: {min(stars):,}  Max: {max(stars):,}  Avg: {sum(stars)//len(stars):,}")

        print(f"\n  {C.BOLD}Repo Size (KB):{C.END}")
        print(f"    Min: {min(sizes):,}  Max: {max(sizes):,}  Avg: {sum(sizes)//len(sizes):,}")

        ci_count = sum(1 for r in repo_list if r.get("has_ci"))
        docker_count = sum(1 for r in repo_list if r.get("has_docker"))
        print(f"\n  {C.BOLD}Infrastructure:{C.END}")
        print(f"    Has CI: {ci_count}/{len(repo_list)}")
        print(f"    Has Docker: {docker_count}/{len(repo_list)}")
    else:
        print(f"  {C.RED}✗ repo_list.json not found{C.END}")

    # ─── Per-Repo Extraction Data ───
    print_section("EXTRACTED DATA (repo_irs)")

    repo_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    print(f"  Repos with extracted data: {C.BOLD}{len(repo_dirs)}{C.END}")

    total_diagrams = 0
    total_files = 0
    total_warnings = 0
    completeness = {"repo_ir": 0, "spec": 0, "teacher_plan": 0, "diagrams": 0}

    for repo_dir in repo_dirs:
        print(f"\n  {C.BOLD}{'─' * 50}{C.END}")
        print(f"  {C.BOLD}📁 {repo_dir.name}{C.END}")

        # Check what files exist
        ir_path = repo_dir / "repo_ir.json"
        spec_path = repo_dir / "spec.json"
        plan_path = repo_dir / "teacher_plan.json"
        diag_dir = repo_dir / "diagrams"

        has_ir = ir_path.exists()
        has_spec = spec_path.exists()
        has_plan = plan_path.exists()
        has_diags = diag_dir.exists() and any(diag_dir.iterdir()) if diag_dir.exists() else False

        status = lambda ok: f"{C.GREEN}✓{C.END}" if ok else f"{C.RED}✗{C.END}"
        print(f"    {status(has_ir)} repo_ir.json ({fmt_size(ir_path.stat().st_size) if has_ir else 'missing'})")
        print(f"    {status(has_spec)} spec.json ({fmt_size(spec_path.stat().st_size) if has_spec else 'missing'})")
        print(f"    {status(has_plan)} teacher_plan.json ({fmt_size(plan_path.stat().st_size) if has_plan else 'missing'})")

        if has_ir:
            completeness["repo_ir"] += 1
        if has_spec:
            completeness["spec"] += 1
        if has_plan:
            completeness["teacher_plan"] += 1

        # RepoIR details
        if has_ir:
            ir = load_json(ir_path)
            if ir:
                meta = ir.get("repo_metadata", {})
                manifest = ir.get("file_manifest", [])
                diagrams = ir.get("diagram_paths", [])
                deps = ir.get("dependencies", [])
                routes = ir.get("api_routes", [])
                models = ir.get("data_models", [])

                print(f"\n    {C.BOLD}RepoIR Summary:{C.END}")
                print(f"      Language: {meta.get('primary_language', '?')}")
                print(f"      LOC: {meta.get('total_loc', 0):,}")
                print(f"      Stars: {meta.get('star_count', 0):,}")
                print(f"      Scale tier: {meta.get('scale_tier', '?')}")
                print(f"      Files in manifest: {len(manifest)}")
                print(f"      Dependencies: {len(deps)}")
                print(f"      API routes: {len(routes)}")
                print(f"      Data models: {len(models)}")
                print(f"      Diagram paths: {len(diagrams)}")

                total_files += len(manifest)
                total_diagrams += len(diagrams)

                if diagrams:
                    completeness["diagrams"] += 1

                # Diagram file check
                if has_diags:
                    diag_files = list(diag_dir.glob("*"))
                    actual_imgs = [f for f in diag_files if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp")]
                    total_size = sum(f.stat().st_size for f in actual_imgs)
                    print(f"      Diagram files on disk: {len(actual_imgs)} ({fmt_size(total_size)})")
                elif diagrams:
                    print(f"      {C.YELLOW}⚠ diagram_paths listed but diagrams/ dir missing or empty{C.END}")

                # Infra
                infra = ir.get("infrastructure", {})
                ci = infra.get("ci_cd")
                deploy = infra.get("deployment_files", [])
                if ci or deploy:
                    print(f"      CI/CD: {ci or 'none'} ({len(deploy)} workflow files)")

                # Quality checks
                ir_warnings = check_repo_ir(ir)
                print_warnings(ir_warnings, "RepoIR quality")
                total_warnings += len(ir_warnings)

        # Spec details
        if has_spec:
            spec = load_json(spec_path)
            if spec:
                reqs = spec.get("functional_requirements", [])
                print(f"\n    {C.BOLD}Spec:{C.END}")
                print(f"      Feature: {spec.get('feature_name', '?')}")
                print(f"      Requirements: {len(reqs)}")
                print(f"      Scale tier: {spec.get('scale_tier', '?')}")

                spec_warnings = check_spec(spec)
                print_warnings(spec_warnings, "Spec quality")
                total_warnings += len(spec_warnings)

        # Teacher plan details
        if has_plan:
            plan = load_json(plan_path)
            if plan:
                decisions = plan.get("architecture_decisions", [])
                tickets = plan.get("tickets", [])
                tech = plan.get("technology_choices", [])

                # Compute RGS if we have both plan and manifest
                rgs = 0.0
                if has_ir and ir:
                    manifest_set = set(ir.get("file_manifest", []))
                    all_paths = set()
                    for d in decisions:
                        all_paths.update(d.get("files_affected", []))
                    for t in tickets:
                        all_paths.update(t.get("files_to_modify", []))
                        all_paths.update(t.get("files_to_create", []))
                    existing = [p for p in all_paths if p in manifest_set or p.rstrip(" (new)") in manifest_set]
                    if all_paths:
                        rgs = len(existing) / len(all_paths)

                print(f"\n    {C.BOLD}Teacher Plan:{C.END}")
                print(f"      Architecture decisions: {len(decisions)}")
                print(f"      Tickets: {len(tickets)}")
                print(f"      Technology choices: {len(tech)}")
                print(f"      File paths referenced: {len(all_paths)}")
                print(f"      RGS (Repo Grounding Score): {C.BOLD}{rgs:.2f}{C.END}")

                plan_warnings = check_teacher_plan(plan)
                print_warnings(plan_warnings, "Plan quality")
                total_warnings += len(plan_warnings)

    # ─── Aggregate Summary ───
    print_section("AGGREGATE SUMMARY")
    n = len(repo_dirs)
    print(f"  Repos processed:      {C.BOLD}{n}{C.END}")
    print(f"  With repo_ir.json:    {completeness['repo_ir']}/{n}")
    print(f"  With spec.json:       {completeness['spec']}/{n}")
    print(f"  With teacher_plan:    {completeness['teacher_plan']}/{n}")
    print(f"  With diagrams:        {completeness['diagrams']}/{n}")
    print(f"  Total files indexed:  {total_files:,}")
    print(f"  Total diagram paths:  {total_diagrams}")
    print(f"  Total warnings:       {C.YELLOW if total_warnings > 0 else C.GREEN}{total_warnings}{C.END}")

    # ─── Data Quality Flags ───
    print_section("DATA QUALITY FLAGS FOR FULL SCRAPE")
    issues = []

    if repo_list:
        # All repos are the same language
        if len(langs) == 1:
            issues.append(f"All repos are {list(langs.keys())[0]} — need language diversity")
        # All repos are mega-popular (not representative)
        if all(s > 100000 for s in stars):
            issues.append("All repos have >100K stars — may not be representative of real projects")
        # No contributors enriched
        if all(r.get("num_contributors", 0) == 0 for r in repo_list):
            issues.append("num_contributors=0 for all repos — GitHub enrichment may be broken")
        # No CI/Docker
        if ci_count == 0:
            issues.append("No repos have CI detected — consider scraping repos with CI")
        # Missing language breakdown
        if all(not r.get("languages") for r in repo_list):
            issues.append("languages dict empty for all repos — language breakdown not populated")
        # Scale tier distribution
        if len(tiers) < 2:
            issues.append(f"Only '{list(tiers.keys())[0]}' scale tier — need tier diversity")
        # Content-only repos (low LOC, no code structure)
        content_repos = [d.name for d in repo_dirs if (load_json(d / "repo_ir.json") or {}).get("repo_metadata", {}).get("total_loc", 0) < 2000]
        if content_repos:
            issues.append(f"Low-code repos (likely content/list repos): {', '.join(content_repos)}")

    if not issues:
        print(f"  {C.GREEN}✓ No critical issues found{C.END}")
    else:
        for issue in issues:
            print(f"  {C.YELLOW}⚠ {issue}{C.END}")

    print(f"\n  {C.BOLD}Recommendations for full scrape:{C.END}")
    print(f"    1. Add multi-language repos (JS/TS, Go, Rust, Java)")
    print(f"    2. Include mid-tier repos (1K-50K stars) for realistic training data")
    print(f"    3. Target repos with actual code structure (APIs, models, infra)")
    print(f"    4. Ensure contributor enrichment works (num_contributors > 0)")
    print(f"    5. Balance scale tiers: hobby, startup, growth, enterprise")
    print()


if __name__ == "__main__":
    main()
