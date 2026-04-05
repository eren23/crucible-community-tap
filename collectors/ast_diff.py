"""AST diff analysis for rich action vectors.

Computes a 15-dim action vector that summarizes the structural difference
between two Python source files at the AST level.

Dimensions:
    0-2:   Edit type (one-hot): ADD / DELETE / MODIFY
    3-5:   Scope (one-hot): function / class / module
    6:     Location — normalized position of first change
    7:     Structural complexity delta (normalized node count change)
    8:     Added functions (count, capped at 5 and normalized)
    9:     Removed functions (count, capped at 5 and normalized)
    10:    Modified functions (count, capped at 10 and normalized)
    11:    Added imports (count, capped at 5 and normalized)
    12:    Control flow changes (for/while/if/try added or removed, normalized)
    13:    Decorator/class changes (count, normalized)
    14:    Size ratio (len(new) / max(len(old), 1), capped at 3.0 and normalized to [0,1])

This vector is interpretable — each dimension has a clear structural meaning.
It enables the "refactoring compass": inverting the predictor to find what
action would move code to a desired state.

Usage::

    action = compute_rich_action(old_source, new_source)
    # action.shape == (15,)
"""
from __future__ import annotations

import ast
from typing import Any

import numpy as np


ACTION_DIM_RICH = 15


def compute_rich_action(old_source: str, new_source: str) -> np.ndarray:
    """Compute a 15-dim structural action vector from before/after Python code.

    Falls back gracefully on parse errors — produces a valid vector in all cases.
    """
    action = np.zeros(ACTION_DIM_RICH, dtype=np.float32)

    old_clean = (old_source or "").strip()
    new_clean = (new_source or "").strip()

    # --- Dims 0-2: Edit type (one-hot) ---
    if not old_clean:
        action[0] = 1.0  # ADD
    elif not new_clean:
        action[1] = 1.0  # DELETE
    else:
        action[2] = 1.0  # MODIFY

    # Parse ASTs (graceful on error)
    old_tree = _safe_parse(old_clean)
    new_tree = _safe_parse(new_clean)

    old_info = _extract_info(old_tree) if old_tree else _empty_info()
    new_info = _extract_info(new_tree) if new_tree else _empty_info()

    # --- Dims 3-5: Scope (one-hot) ---
    scope = _detect_scope_from_info(old_info, new_info)
    action[3 + scope] = 1.0

    # --- Dim 6: Location ---
    action[6] = _estimate_location(old_source or "", new_source or "")

    # --- Dim 7: Structural complexity delta ---
    # Normalized change in total AST node count
    delta = new_info["node_count"] - old_info["node_count"]
    max_count = max(old_info["node_count"], new_info["node_count"], 1)
    action[7] = np.clip(delta / max_count, -1.0, 1.0) * 0.5 + 0.5  # map to [0, 1]

    # --- Dim 8: Added functions ---
    added_funcs = new_info["func_names"] - old_info["func_names"]
    action[8] = min(len(added_funcs), 5) / 5.0

    # --- Dim 9: Removed functions ---
    removed_funcs = old_info["func_names"] - new_info["func_names"]
    action[9] = min(len(removed_funcs), 5) / 5.0

    # --- Dim 10: Modified functions ---
    common_funcs = old_info["func_names"] & new_info["func_names"]
    modified = sum(
        1 for f in common_funcs
        if old_info["func_sizes"].get(f, 0) != new_info["func_sizes"].get(f, 0)
    )
    action[10] = min(modified, 10) / 10.0

    # --- Dim 11: Added imports ---
    added_imports = new_info["imports"] - old_info["imports"]
    action[11] = min(len(added_imports), 5) / 5.0

    # --- Dim 12: Control flow changes ---
    cf_delta = abs(new_info["control_flow_count"] - old_info["control_flow_count"])
    action[12] = min(cf_delta, 10) / 10.0

    # --- Dim 13: Decorator/class changes ---
    class_delta = abs(len(new_info["class_names"]) - len(old_info["class_names"]))
    deco_delta = abs(new_info["decorator_count"] - old_info["decorator_count"])
    action[13] = min(class_delta + deco_delta, 5) / 5.0

    # --- Dim 14: Size ratio ---
    old_len = max(len(old_clean), 1)
    ratio = len(new_clean) / old_len
    action[14] = min(ratio, 3.0) / 3.0  # normalize to [0, 1], cap at 3x

    return action


# ---------------------------------------------------------------------------
# AST analysis helpers
# ---------------------------------------------------------------------------

def _safe_parse(source: str) -> ast.AST | None:
    """Parse Python source, returning None on failure."""
    if not source:
        return None
    try:
        return ast.parse(source.replace("\x00", ""))
    except (SyntaxError, ValueError, RecursionError):
        return None


def _empty_info() -> dict[str, Any]:
    """Return empty info dict for unparseable code."""
    return {
        "node_count": 0,
        "func_names": set(),
        "func_sizes": {},
        "class_names": set(),
        "imports": set(),
        "control_flow_count": 0,
        "decorator_count": 0,
    }


_CONTROL_FLOW_TYPES = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try, ast.With, ast.AsyncWith)


def _extract_info(tree: ast.AST) -> dict[str, Any]:
    """Extract structural information from a parsed AST."""
    func_names: set[str] = set()
    func_sizes: dict[str, int] = {}
    class_names: set[str] = set()
    imports: set[str] = set()
    control_flow_count = 0
    decorator_count = 0
    node_count = 0

    for node in ast.walk(tree):
        node_count += 1

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_names.add(node.name)
            func_sizes[node.name] = sum(1 for _ in ast.walk(node))
            decorator_count += len(node.decorator_list)

        elif isinstance(node, ast.ClassDef):
            class_names.add(node.name)
            decorator_count += len(node.decorator_list)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

        elif isinstance(node, _CONTROL_FLOW_TYPES):
            control_flow_count += 1

    return {
        "node_count": node_count,
        "func_names": func_names,
        "func_sizes": func_sizes,
        "class_names": class_names,
        "imports": imports,
        "control_flow_count": control_flow_count,
        "decorator_count": decorator_count,
    }


def _detect_scope_from_info(
    old_info: dict[str, Any], new_info: dict[str, Any]
) -> int:
    """Detect scope from pre-extracted info. 0=function, 1=class, 2=module."""
    if old_info["func_names"] != new_info["func_names"]:
        return 0
    if old_info["class_names"] != new_info["class_names"]:
        return 1
    # Check modified function bodies
    common = old_info["func_names"] & new_info["func_names"]
    for f in common:
        if old_info["func_sizes"].get(f, 0) != new_info["func_sizes"].get(f, 0):
            return 0
    return 2


def _estimate_location(old_source: str, new_source: str) -> float:
    """Normalized position (0-1) of first difference."""
    old_lines = old_source.splitlines() if old_source else []
    new_lines = new_source.splitlines() if new_source else []
    total = max(len(old_lines), len(new_lines), 1)

    for i, (a, b) in enumerate(zip(old_lines, new_lines)):
        if a != b:
            return i / total

    return min(len(old_lines), len(new_lines)) / total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_action_dim() -> int:
    """Return the rich action vector dimensionality."""
    return ACTION_DIM_RICH
