"""ORM model extractor: detect database models from Django, SQLAlchemy, Prisma."""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build"}

# Django field types
DJANGO_FIELDS = {
    "CharField",
    "TextField",
    "IntegerField",
    "FloatField",
    "DecimalField",
    "BooleanField",
    "DateField",
    "DateTimeField",
    "TimeField",
    "EmailField",
    "URLField",
    "UUIDField",
    "FileField",
    "ImageField",
    "JSONField",
    "SlugField",
    "AutoField",
    "BigAutoField",
    "SmallAutoField",
    "BigIntegerField",
    "SmallIntegerField",
    "PositiveIntegerField",
    "BinaryField",
    "DurationField",
    "GenericIPAddressField",
}

DJANGO_RELATION_FIELDS = {"ForeignKey", "OneToOneField", "ManyToManyField"}

# SQLAlchemy column types
SQLA_TYPES = {
    "Integer",
    "String",
    "Text",
    "Float",
    "Numeric",
    "Boolean",
    "Date",
    "DateTime",
    "Time",
    "LargeBinary",
    "Enum",
    "JSON",
    "ARRAY",
    "UUID",
}


def _extract_django_models(file_path: str, rel_path: str) -> list[dict]:
    """Extract Django models via AST."""
    models = []
    try:
        with open(file_path, "r", errors="ignore") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError):
        return models

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check if inherits from models.Model or Model
        is_django_model = False
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Model":
                is_django_model = True
            elif isinstance(base, ast.Name) and base.id in ("Model", "AbstractUser", "AbstractBaseUser"):
                is_django_model = True
        if not is_django_model:
            continue

        fields = []
        relationships = []

        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    field_name = target.id
                    if isinstance(item.value, ast.Call):
                        func = item.value.func
                        field_type = _get_call_name(func)
                        if field_type:
                            short_type = field_type.split(".")[-1]
                            if short_type in DJANGO_RELATION_FIELDS:
                                # Extract related model
                                if item.value.args:
                                    related = _get_str_or_name(item.value.args[0])
                                    if related:
                                        relationships.append(f"{short_type} -> {related}")
                            fields.append({
                                "name": field_name,
                                "field_type": short_type,
                                "constraints": _extract_django_field_constraints(item.value),
                            })

        models.append({
            "name": node.name,
            "file_path": rel_path,
            "fields": fields,
            "orm": "django",
            "relationships": relationships,
        })

    return models


def _extract_sqlalchemy_models(file_path: str, rel_path: str) -> list[dict]:
    """Extract SQLAlchemy models via AST."""
    models = []
    try:
        with open(file_path, "r", errors="ignore") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError):
        return models

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check for SQLAlchemy base classes
        # Exclude Pydantic BaseModel and similar non-ORM bases
        SQLA_BASES = {"Base", "DeclarativeBase", "db.Model", "SQLModel"}
        NON_SQLA_BASES = {"BaseModel", "BaseSettings", "BaseConfig"}
        is_sqla = False
        for base in node.bases:
            name = _get_call_name(base) or ""
            if name in NON_SQLA_BASES:
                is_sqla = False
                break
            if name in SQLA_BASES:
                is_sqla = True
        if not is_sqla:
            continue

        fields = []
        relationships = []

        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    field_name = target.id
                    if field_name == "__tablename__":
                        continue
                    if isinstance(item.value, ast.Call):
                        func_name = _get_call_name(item.value.func) or ""
                        if "Column" in func_name:
                            col_type = "unknown"
                            if item.value.args:
                                col_type = _get_call_name(item.value.args[0]) or "unknown"
                            fields.append({
                                "name": field_name,
                                "field_type": col_type.split(".")[-1],
                                "constraints": [],
                            })
                        elif "relationship" in func_name:
                            if item.value.args:
                                related = _get_str_or_name(item.value.args[0])
                                if related:
                                    relationships.append(f"relationship -> {related}")

        models.append({
            "name": node.name,
            "file_path": rel_path,
            "fields": fields,
            "orm": "sqlalchemy",
            "relationships": relationships,
        })

    return models


def _extract_prisma_models(file_path: str, rel_path: str) -> list[dict]:
    """Extract Prisma models from schema.prisma."""
    models = []
    try:
        with open(file_path, "r", errors="ignore") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return models

    # Match: model User { ... }
    model_pattern = r"model\s+(\w+)\s*\{([^}]+)\}"
    for match in re.finditer(model_pattern, content):
        name = match.group(1)
        body = match.group(2)
        fields = []
        relationships = []

        for line in body.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("@@"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                field_name = parts[0]
                field_type = parts[1]
                constraints = parts[2:] if len(parts) > 2 else []
                # Detect relations (type starts with uppercase and not a primitive)
                base_type = field_type.rstrip("?[]")
                if base_type[0:1].isupper() and base_type not in ("String", "Int", "Float", "Boolean", "DateTime", "Json", "Bytes", "Decimal", "BigInt"):
                    relationships.append(f"relation -> {base_type}")
                fields.append({
                    "name": field_name,
                    "field_type": field_type,
                    "constraints": [c for c in constraints if c.startswith("@")],
                })

        models.append({
            "name": name,
            "file_path": rel_path,
            "fields": fields,
            "orm": "prisma",
            "relationships": relationships,
        })

    return models


def _get_call_name(node: ast.expr) -> str | None:
    """Get the dotted name from an AST call/attribute node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _get_call_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    if isinstance(node, ast.Call):
        return _get_call_name(node.func)
    return None


def _get_str_or_name(node: ast.expr) -> str | None:
    """Get string constant or Name id from AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _extract_django_field_constraints(call_node: ast.Call) -> list[str]:
    """Extract constraint keywords from a Django field call."""
    constraints = []
    for kw in call_node.keywords:
        if kw.arg in ("unique", "null", "blank", "primary_key", "db_index"):
            if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                constraints.append(kw.arg)
        elif kw.arg == "max_length" and isinstance(kw.value, ast.Constant):
            constraints.append(f"max_length={kw.value.value}")
    return constraints


def extract_data_models(repo_path: str) -> list[dict]:
    """Extract all ORM/database models from a repository."""
    repo = Path(repo_path)
    all_models: list[dict] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            fpath = os.path.join(root, fname)
            rel_path = str(Path(fpath).relative_to(repo))

            if fname.endswith(".py"):
                # Check for Django models
                if "models" in fname or "model" in fname:
                    all_models.extend(_extract_django_models(fpath, rel_path))
                # Check for SQLAlchemy (could be in any .py file)
                all_models.extend(_extract_sqlalchemy_models(fpath, rel_path))

            elif fname == "schema.prisma":
                all_models.extend(_extract_prisma_models(fpath, rel_path))

    # Deduplicate by (name, file_path)
    seen: set[tuple] = set()
    unique: list[dict] = []
    for model in all_models:
        key = (model["name"], model["file_path"])
        if key not in seen:
            seen.add(key)
            unique.append(model)

    return unique
