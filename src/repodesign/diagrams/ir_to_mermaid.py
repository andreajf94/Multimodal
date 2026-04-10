"""Convert RepoIR fields to Mermaid diagram strings.

Pure functions — no I/O. Each returns a list of Mermaid strings (one per
chunk), or an empty list if there is nothing to render.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# API Routes → Mermaid flowchart
# ---------------------------------------------------------------------------

_METHOD_COLORS = {
    "get":    "fill:#61affe,color:#fff",
    "post":   "fill:#49cc90,color:#fff",
    "put":    "fill:#fca130,color:#fff",
    "delete": "fill:#f93e3e,color:#fff",
    "patch":  "fill:#50e3c2,color:#fff",
}
_MAX_ROUTES_PER_CHART = 12


def api_routes_to_mermaid(routes: list[dict]) -> list[str]:
    """Convert a list of APIRoute dicts to Mermaid LR flowchart strings.

    Deduplicates by (path, method). Chunks into groups of MAX_ROUTES_PER_CHART.
    Returns a list of Mermaid strings (one per chart), or empty list.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for r in routes:
        key = (r.get("path", ""), r.get("method", "").upper())
        if key not in seen:
            seen.add(key)
            unique.append(r)

    if not unique:
        return []

    # Chunk into groups
    chunks = [unique[i:i + _MAX_ROUTES_PER_CHART]
              for i in range(0, len(unique), _MAX_ROUTES_PER_CHART)]

    results = []
    for chunk in chunks:
        mmd = _render_route_chart(chunk)
        if mmd:
            results.append(mmd)
    return results


def _render_route_chart(routes: list[dict]) -> str:
    """Render a single flowchart from a list of routes."""
    groups: dict[str, list[dict]] = {}
    for r in routes:
        path = r.get("path", "/")
        parts = path.strip("/").split("/")
        prefix = "/" + parts[0] if parts and parts[0] else "/"
        groups.setdefault(prefix, []).append(r)

    lines = ["graph LR"]

    for g_idx, (prefix, g_routes) in enumerate(groups.items()):
        sg_label = prefix.replace('"', "'")
        lines.append(f'    subgraph sg{g_idx}["{sg_label}"]')
        for r_idx, r in enumerate(g_routes):
            node_id = f"n{g_idx}_{r_idx}"
            method = r.get("method", "GET").upper()
            path = r.get("path", "")
            handler_file = r.get("handler_file", "").replace("\\", "/")
            filename = handler_file.split("/")[-1] if handler_file else ""
            handler_fn = r.get("handler_function") or ""

            top = f"{method} {path}"
            bottom = filename + (f"::{handler_fn}" if handler_fn else "")
            label = f"{top}<br/>{bottom}" if bottom else top
            label = label.replace('"', "'")

            css = method.lower() if method.lower() in _METHOD_COLORS else "other"
            lines.append(f'        {node_id}["{label}"]:::{css}')
        lines.append("    end")

    for method, style in _METHOD_COLORS.items():
        lines.append(f"    classDef {method} {style}")
    lines.append("    classDef other fill:#aaaaaa,color:#fff")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data Models → Mermaid ER diagram
# ---------------------------------------------------------------------------

_MAX_MODELS_PER_CHART = 8
_MAX_FIELDS_PER_MODEL = 10

_MERMAID_UNSAFE = re.compile(r"[^A-Za-z0-9_]")


def _clean_type(raw: str) -> str:
    """Sanitise a Prisma/SQLAlchemy type for Mermaid ER syntax."""
    t = raw.replace("?", "").replace("[]", "").strip()
    t = _MERMAID_UNSAFE.sub("_", t)
    return t or "string"


def _clean_name(raw: str) -> str:
    """Sanitise a field or model name for Mermaid ER syntax."""
    return _MERMAID_UNSAFE.sub("_", raw) or "field"


def _build_fk_graph(by_name: dict[str, dict]) -> dict[str, set[str]]:
    """Build a bidirectional FK adjacency map from model fields.

    If model A has a field ``fooId`` and model ``Foo`` exists, both
    ``A -> Foo`` and ``Foo -> A`` edges are recorded.
    """
    name_lower_map = {n.lower(): n for n in by_name}
    graph: dict[str, set[str]] = {n: set() for n in by_name}

    for name, m in by_name.items():
        for f in m.get("fields", []):
            fname = f.get("name", "")
            ref_base: str | None = None
            if fname.endswith("Id"):
                ref_base = fname[:-2]
            elif fname.endswith("_id"):
                ref_base = fname[:-3]
            if ref_base:
                ref_key = ref_base.lower()
                if ref_key in name_lower_map:
                    target = name_lower_map[ref_key]
                    if target != name:
                        graph[name].add(target)
                        graph[target].add(name)

            # Also check if field type matches a model name (Prisma relation fields)
            ftype = f.get("field_type", "").replace("?", "").replace("[]", "").strip()
            ftype_key = ftype.lower()
            if ftype_key in name_lower_map:
                target = name_lower_map[ftype_key]
                if target != name:
                    graph[name].add(target)
                    graph[target].add(name)

    return graph


def models_mentioned_in_diff(
    all_models: list[dict],
    diff_text: str,
) -> list[dict]:
    """Filter models to those mentioned in the diff, plus their FK neighbors.

    1. Scan the diff for model name occurrences (word-boundary match).
    2. For each matched model, pull in immediate FK neighbors (models linked
       by ``fooId`` fields or typed relation fields).
    3. Return the connected subgraph, capped at MAX_NEIGHBOR_EXPANSION total.

    This ensures diagrams always show a connected cluster with relationship
    lines, rather than isolated single-entity boxes.
    """
    if not diff_text or not all_models:
        return []

    # Deduplicate by name, prefer more fields
    by_name: dict[str, dict] = {}
    for m in all_models:
        name = m.get("name", "").strip()
        if not name or len(name) < 3:
            continue
        existing = by_name.get(name)
        if existing is None or len(m.get("fields", [])) > len(existing.get("fields", [])):
            by_name[name] = m

    # Find models mentioned in the diff
    mentions: dict[str, int] = {}
    for name in by_name:
        pattern = re.compile(r'\b' + re.escape(name) + r'\b')
        count = len(pattern.findall(diff_text))
        if count > 0:
            mentions[name] = count

    if not mentions:
        return []

    # Build FK adjacency graph and expand seeds to neighbors
    fk_graph = _build_fk_graph(by_name)
    seeds = sorted(mentions.keys(), key=lambda n: mentions[n], reverse=True)

    selected: dict[str, int] = {}  # name -> priority (lower = more important)
    for priority, name in enumerate(seeds):
        if name not in selected:
            selected[name] = priority

    # Expand: add FK neighbors of seed models
    for name in list(seeds):
        for neighbor in fk_graph.get(name, set()):
            if neighbor not in selected:
                selected[neighbor] = len(selected)

    # Cap total and filter to models with fields
    ranked_names = sorted(selected.keys(), key=lambda n: selected[n])
    ranked_names = ranked_names[:_MAX_NEIGHBOR_EXPANSION]
    return [by_name[n] for n in ranked_names if by_name[n].get("fields")]


_MAX_NEIGHBOR_EXPANSION = 16  # max models after FK expansion (2 charts × 8)


def data_models_to_er(models: list[dict]) -> list[str]:
    """Convert a list of DataModel dicts to Mermaid erDiagram strings.

    Deduplicates model names, preferring instances with more fields.
    Chunks into groups of MAX_MODELS_PER_CHART.
    Returns a list of Mermaid strings (one per chart), or empty list.
    """
    if not models:
        return []

    # Deduplicate by name
    by_name: dict[str, dict] = {}
    for m in models:
        name = m.get("name", "").strip()
        if not name:
            continue
        existing = by_name.get(name)
        if existing is None or len(m.get("fields", [])) > len(existing.get("fields", [])):
            by_name[name] = m

    with_fields = [m for m in by_name.values() if m.get("fields")]
    if not with_fields:
        return []

    ranked = sorted(with_fields, key=lambda m: len(m.get("fields", [])), reverse=True)

    # Chunk into groups
    chunks = [ranked[i:i + _MAX_MODELS_PER_CHART]
              for i in range(0, len(ranked), _MAX_MODELS_PER_CHART)]

    results = []
    for chunk in chunks:
        mmd = _render_er_chart(chunk)
        if mmd:
            results.append(mmd)
    return results


def _render_er_chart(models: list[dict]) -> str | None:
    """Render a single ER diagram from a list of models.

    Returns None if fewer than 2 models (single-entity diagrams are useless).
    """
    if len(models) < 2:
        return None
    clean_model_names: dict[str, str] = {
        m["name"].lower(): m["name"] for m in models
    }

    relationships: list[tuple[str, str]] = []
    lines = ["erDiagram"]

    for m in models:
        raw_name = m.get("name", "")
        entity = _clean_name(raw_name)
        fields = m.get("fields", [])[:_MAX_FIELDS_PER_MODEL]

        lines.append(f"    {entity} {{")
        for f in fields:
            fname = _clean_name(f.get("name", "field"))
            ftype = _clean_type(f.get("field_type", "string"))
            lines.append(f"        {ftype} {fname}")

            ref_base: str | None = None
            if fname.endswith("Id"):
                ref_base = fname[:-2]
            elif fname.endswith("_id"):
                ref_base = fname[:-3]

            if ref_base:
                ref_key = ref_base.lower()
                if ref_key in clean_model_names and clean_model_names[ref_key] != raw_name:
                    pair = (entity, _clean_name(clean_model_names[ref_key]))
                    if pair not in relationships:
                        relationships.append(pair)

        lines.append("    }")

    for many, one in relationships:
        lines.append(f'    {many} }}o--|| {one} : ""')

    return "\n".join(lines)
