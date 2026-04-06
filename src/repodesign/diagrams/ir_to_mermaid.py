"""Convert RepoIR fields to Mermaid diagram strings.

Pure functions — no I/O. Each returns a Mermaid string or None if there is
nothing to render (empty input, all duplicates filtered, etc.).
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
_MAX_ROUTES = 20


def api_routes_to_mermaid(routes: list[dict]) -> str | None:
    """Convert a list of APIRoute dicts to a Mermaid LR flowchart string.

    Deduplicates by (path, method) — keeps the first occurrence of each pair.
    Groups routes into subgraphs by the first path segment.
    Caps at MAX_ROUTES unique routes.
    Returns None if there is nothing to render.
    """
    # Deduplicate by (path, method)
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for r in routes:
        key = (r.get("path", ""), r.get("method", "").upper())
        if key not in seen:
            seen.add(key)
            unique.append(r)

    if not unique:
        return None

    unique = unique[:_MAX_ROUTES]

    # Group by first path segment
    groups: dict[str, list[dict]] = {}
    for r in unique:
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

            # Build two-line label: "METHOD /path<br/>file::fn"
            top = f"{method} {path}"
            bottom = filename + (f"::{handler_fn}" if handler_fn else "")
            label = f"{top}<br/>{bottom}" if bottom else top
            label = label.replace('"', "'")

            css = method.lower() if method.lower() in _METHOD_COLORS else "other"
            lines.append(f'        {node_id}["{label}"]:::{css}')
        lines.append("    end")

    # classDef declarations
    for method, style in _METHOD_COLORS.items():
        lines.append(f"    classDef {method} {style}")
    lines.append("    classDef other fill:#aaaaaa,color:#fff")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data Models → Mermaid ER diagram
# ---------------------------------------------------------------------------

_MAX_MODELS = 15
_MAX_FIELDS_PER_MODEL = 12

# Characters not valid in Mermaid ER field types / names
_MERMAID_UNSAFE = re.compile(r"[^A-Za-z0-9_]")


def _clean_type(raw: str) -> str:
    """Sanitise a Prisma/SQLAlchemy type for Mermaid ER syntax.

    Strips optional markers (?), array markers ([]), and anything else that
    Mermaid would reject.  Falls back to 'string' if nothing remains.
    """
    t = raw.replace("?", "").replace("[]", "").strip()
    t = _MERMAID_UNSAFE.sub("_", t)
    return t or "string"


def _clean_name(raw: str) -> str:
    """Sanitise a field or model name for Mermaid ER syntax."""
    return _MERMAID_UNSAFE.sub("_", raw) or "field"


def data_models_to_er(models: list[dict]) -> str | None:
    """Convert a list of DataModel dicts to a Mermaid erDiagram string.

    Deduplicates model names, preferring instances with more fields.
    Infers relationships from fields whose name ends in 'Id' or '_id' and
    whose prefix matches another model name.
    Returns None if there is nothing to render.
    """
    if not models:
        return None

    # Deduplicate by name — keep the version with the most fields
    by_name: dict[str, dict] = {}
    for m in models:
        name = m.get("name", "").strip()
        if not name:
            continue
        existing = by_name.get(name)
        if existing is None or len(m.get("fields", [])) > len(existing.get("fields", [])):
            by_name[name] = m

    if not by_name:
        return None

    # Drop models with no fields if any field-populated models exist; return
    # None if there is nothing substantive to render.
    with_fields = [m for m in by_name.values() if m.get("fields")]
    if not with_fields:
        return None

    ranked = sorted(with_fields, key=lambda m: len(m.get("fields", [])), reverse=True)
    ranked = ranked[:_MAX_MODELS]

    # Build a lookup of clean model names for relationship inference
    clean_model_names: dict[str, str] = {
        m["name"].lower(): m["name"] for m in ranked
    }

    relationships: list[tuple[str, str]] = []  # (many_side, one_side)

    lines = ["erDiagram"]

    for m in ranked:
        raw_name = m.get("name", "")
        entity = _clean_name(raw_name)
        fields = m.get("fields", [])[:_MAX_FIELDS_PER_MODEL]

        lines.append(f"    {entity} {{")
        for f in fields:
            fname = _clean_name(f.get("name", "field"))
            ftype = _clean_type(f.get("field_type", "string"))
            lines.append(f"        {ftype} {fname}")

            # Infer FK relationship: fooId → Foo, foo_id → Foo
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
