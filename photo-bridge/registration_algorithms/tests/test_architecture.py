"""Architecture guard: production must not import the research package."""
from __future__ import annotations

import ast
from pathlib import Path


def test_production_does_not_import_research():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        if "research" in path.parts or "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imported = None
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""]
            if imported and any("research" in name for name in imported):
                offenders.append((str(path.relative_to(root)), ast.unparse(node)))
    assert not offenders, offenders