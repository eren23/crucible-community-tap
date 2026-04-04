"""Python AST tokenizer for code world models.

Converts Python source code into a fixed-length sequence of semantically
meaningful tokens derived from the Abstract Syntax Tree.

Vocabulary layout (~650 tokens):
    0-99:    AST node types (Module, FunctionDef, If, For, Assign, etc.)
    100-611: Identifier hash buckets (512 buckets for names/attributes)
    612-649: Special tokens (PAD, BOS, EOS, UNK, PARSE_ERROR, DEPTH_0..15, operators)

This tokenizer captures code STRUCTURE, not surface syntax. Two programs
with different formatting but identical AST produce identical token sequences.
This is ideal for JEPA-style world models that predict in latent space.

Usage::

    tokens = ast_tokenize("def foo(): return 1", max_len=256)
    # [BOS, FunctionDef_id, DEPTH_0, hash("foo"), Return_id, DEPTH_1,
    #  Constant_id, DEPTH_2, EOS, PAD, PAD, ...]
"""
from __future__ import annotations

import ast
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary: AST node types (0-99)
# ---------------------------------------------------------------------------

# Enumerate all concrete AST node types in a stable order.
# Abstract bases (AST, mod, stmt, expr, etc.) are excluded.
_ABSTRACT_BASES = frozenset({
    "AST", "mod", "stmt", "expr", "expr_context", "boolop", "operator",
    "unaryop", "cmpop", "comprehension", "excepthandler", "arguments",
    "arg", "keyword", "alias", "withitem", "match_case", "pattern",
    "type_ignore", "type_param",
})


def _get_concrete_node_types() -> list[str]:
    """Collect all concrete AST node type names in sorted order."""
    all_types: set[str] = set()

    def _walk_subclasses(cls: type) -> None:
        name = cls.__name__
        if name not in _ABSTRACT_BASES:
            all_types.add(name)
        for sub in cls.__subclasses__():
            _walk_subclasses(sub)

    _walk_subclasses(ast.AST)
    return sorted(all_types)


_CONCRETE_TYPES = _get_concrete_node_types()
NODE_TYPE_MAP: dict[str, int] = {name: i for i, name in enumerate(_CONCRETE_TYPES)}
NUM_NODE_TYPES = len(_CONCRETE_TYPES)  # ~90-100 depending on Python version

# ---------------------------------------------------------------------------
# Vocabulary: Identifier hash buckets (100-611)
# ---------------------------------------------------------------------------

_IDENT_OFFSET = 100
_IDENT_BUCKETS = 512


def _ident_token(name: str) -> int:
    """Map an identifier string to a hash bucket token ID."""
    return _IDENT_OFFSET + (hash(name) % _IDENT_BUCKETS)


# ---------------------------------------------------------------------------
# Vocabulary: Special tokens (612+)
# ---------------------------------------------------------------------------

_SPECIAL_OFFSET = _IDENT_OFFSET + _IDENT_BUCKETS  # 612

PAD = _SPECIAL_OFFSET + 0          # 612
BOS = _SPECIAL_OFFSET + 1          # 613
EOS = _SPECIAL_OFFSET + 2          # 614
UNK = _SPECIAL_OFFSET + 3          # 615
PARSE_ERROR = _SPECIAL_OFFSET + 4  # 616

# Depth markers: DEPTH_0 through DEPTH_15
_DEPTH_OFFSET = _SPECIAL_OFFSET + 5  # 617
MAX_DEPTH = 15

# Operators start after depth markers
_OP_OFFSET = _DEPTH_OFFSET + MAX_DEPTH + 1  # 633

# Map operator AST type names to token IDs
_OP_NAMES = [
    # Binary operators
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow",
    "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "MatMult",
    # Unary operators
    "Invert", "Not", "UAdd", "USub",
    # Boolean operators
    "And", "Or",
    # Comparison operators
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
]

OP_MAP: dict[str, int] = {name: _OP_OFFSET + i for i, name in enumerate(_OP_NAMES)}

# DFS structure tokens (after operators)
_STRUCT_OFFSET = _OP_OFFSET + len(_OP_NAMES)
OPEN_BRACKET = _STRUCT_OFFSET        # Start of compound node children
CLOSE_BRACKET = _STRUCT_OFFSET + 1   # End of compound node children

VOCAB_SIZE = _STRUCT_OFFSET + 2  # ~665


# ---------------------------------------------------------------------------
# Depth computation
# ---------------------------------------------------------------------------

def _annotate_depths(tree: ast.AST) -> dict[int, int]:
    """Map node id -> depth via BFS from root."""
    depths: dict[int, int] = {id(tree): 0}

    def _visit(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            depths[id(child)] = depth + 1
            _visit(child, depth + 1)

    _visit(tree, 0)
    return depths


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def ast_tokenize(source: str, max_len: int = 512) -> np.ndarray:
    """Tokenize Python source code into AST node token sequence.

    Parameters
    ----------
    source:
        Python source code string. Empty string produces [BOS, EOS, PAD...].
    max_len:
        Fixed output length. Sequences are truncated or padded.

    Returns
    -------
    np.ndarray of shape (max_len,) with dtype uint16.
    """
    if not source or not source.strip():
        tokens = [BOS, EOS]
        tokens += [PAD] * (max_len - len(tokens))
        return np.array(tokens[:max_len], dtype=np.uint16)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        tokens = [BOS, PARSE_ERROR, EOS]
        tokens += [PAD] * (max_len - len(tokens))
        return np.array(tokens[:max_len], dtype=np.uint16)

    depths = _annotate_depths(tree)
    tokens: list[int] = [BOS]

    # BFS walk — captures tree structure top-down
    for node in ast.walk(tree):
        node_name = type(node).__name__

        # Node type token
        node_id = NODE_TYPE_MAP.get(node_name)
        if node_id is not None:
            tokens.append(node_id)
        else:
            tokens.append(UNK)

        # Depth marker
        depth = min(depths.get(id(node), 0), MAX_DEPTH)
        tokens.append(_DEPTH_OFFSET + depth)

        # Identifier tokens for Name and Attribute nodes
        if isinstance(node, ast.Name):
            tokens.append(_ident_token(node.id))
        elif isinstance(node, ast.Attribute):
            tokens.append(_ident_token(node.attr))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            tokens.append(_ident_token(node.name))
        elif isinstance(node, ast.ClassDef):
            tokens.append(_ident_token(node.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            tokens.append(_ident_token(node.module))

        # Operator tokens
        if isinstance(node, ast.BinOp):
            op_name = type(node.op).__name__
            tokens.append(OP_MAP.get(op_name, UNK))
        elif isinstance(node, ast.UnaryOp):
            op_name = type(node.op).__name__
            tokens.append(OP_MAP.get(op_name, UNK))
        elif isinstance(node, ast.BoolOp):
            op_name = type(node.op).__name__
            tokens.append(OP_MAP.get(op_name, UNK))
        elif isinstance(node, ast.Compare) and node.ops:
            # Emit first comparison operator
            op_name = type(node.ops[0]).__name__
            tokens.append(OP_MAP.get(op_name, UNK))

        # Constant value type hint (for Constant nodes)
        if isinstance(node, ast.Constant):
            # Hash the type of the constant for some semantic signal
            val_type = type(node.value).__name__
            tokens.append(_ident_token(f"__const_{val_type}__"))

        # Bail early if we've hit the limit
        if len(tokens) >= max_len - 1:
            break

    tokens.append(EOS)

    # Pad or truncate
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens += [PAD] * (max_len - len(tokens))

    return np.array(tokens, dtype=np.uint16)


def ast_tokenize_dfs(source: str, max_len: int = 512) -> np.ndarray:
    """Tokenize Python source via DFS pre-order traversal (preserves structure).

    Unlike ``ast_tokenize`` (BFS), this uses depth-first traversal with
    OPEN/CLOSE bracket tokens around compound nodes, producing a linearized
    tree that preserves parent-child relationships.

    Example: ``def foo(): return 1`` becomes::

        BOS FunctionDef ident:foo OPEN Return Constant ident:int CLOSE EOS

    The brackets let the transformer reconstruct tree structure — two functions
    with different control flow produce DIFFERENT token sequences (unlike BFS
    which yields the same bag of nodes).
    """
    if not source or not source.strip():
        tokens = [BOS, EOS]
        tokens += [PAD] * (max_len - len(tokens))
        return np.array(tokens[:max_len], dtype=np.uint16)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        tokens = [BOS, PARSE_ERROR, EOS]
        tokens += [PAD] * (max_len - len(tokens))
        return np.array(tokens[:max_len], dtype=np.uint16)

    tokens: list[int] = [BOS]
    _limit = max_len - 1  # reserve space for EOS

    def _visit_dfs(node: ast.AST, depth: int) -> bool:
        """DFS visit. Returns False if token budget exhausted."""
        if len(tokens) >= _limit:
            return False

        node_name = type(node).__name__

        # Node type token
        nid = NODE_TYPE_MAP.get(node_name)
        tokens.append(nid if nid is not None else UNK)

        # Depth marker
        tokens.append(_DEPTH_OFFSET + min(depth, MAX_DEPTH))

        # Identifier / name tokens
        if isinstance(node, ast.Name):
            tokens.append(_ident_token(node.id))
        elif isinstance(node, ast.Attribute):
            tokens.append(_ident_token(node.attr))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            tokens.append(_ident_token(node.name))
        elif isinstance(node, ast.ClassDef):
            tokens.append(_ident_token(node.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            tokens.append(_ident_token(node.module))

        # Operator tokens
        if isinstance(node, ast.BinOp):
            tokens.append(OP_MAP.get(type(node.op).__name__, UNK))
        elif isinstance(node, ast.UnaryOp):
            tokens.append(OP_MAP.get(type(node.op).__name__, UNK))
        elif isinstance(node, ast.BoolOp):
            tokens.append(OP_MAP.get(type(node.op).__name__, UNK))
        elif isinstance(node, ast.Compare) and node.ops:
            tokens.append(OP_MAP.get(type(node.ops[0]).__name__, UNK))

        # Constant type hint
        if isinstance(node, ast.Constant):
            tokens.append(_ident_token(f"__const_{type(node.value).__name__}__"))

        # Recurse into children with OPEN/CLOSE brackets
        children = list(ast.iter_child_nodes(node))
        if children and len(tokens) < _limit:
            tokens.append(OPEN_BRACKET)
            for child in children:
                if not _visit_dfs(child, depth + 1):
                    break
            if len(tokens) < _limit:
                tokens.append(CLOSE_BRACKET)

        return len(tokens) < _limit

    _visit_dfs(tree, 0)
    tokens.append(EOS)

    # Pad or truncate
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens += [PAD] * (max_len - len(tokens))

    return np.array(tokens, dtype=np.uint16)


def decode_tokens(token_ids: np.ndarray) -> list[str]:
    """Decode token IDs back to human-readable names (for debugging).

    Returns a list of strings like ["BOS", "FunctionDef", "DEPTH_0", "hash:42", ...].
    """
    # Build reverse maps
    reverse_node = {v: k for k, v in NODE_TYPE_MAP.items()}
    reverse_op = {v: k for k, v in OP_MAP.items()}

    result: list[str] = []
    for tid in token_ids:
        tid = int(tid)
        if tid == PAD:
            result.append("PAD")
        elif tid == BOS:
            result.append("BOS")
        elif tid == EOS:
            result.append("EOS")
        elif tid == UNK:
            result.append("UNK")
        elif tid == PARSE_ERROR:
            result.append("PARSE_ERROR")
        elif _DEPTH_OFFSET <= tid <= _DEPTH_OFFSET + MAX_DEPTH:
            result.append(f"DEPTH_{tid - _DEPTH_OFFSET}")
        elif tid in reverse_node:
            result.append(reverse_node[tid])
        elif _IDENT_OFFSET <= tid < _IDENT_OFFSET + _IDENT_BUCKETS:
            result.append(f"ident:{tid - _IDENT_OFFSET}")
        elif tid in reverse_op:
            result.append(reverse_op[tid])
        else:
            result.append(f"?{tid}")
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_vocab_size() -> int:
    """Return the total vocabulary size."""
    return VOCAB_SIZE


def get_special_tokens() -> dict[str, int]:
    """Return dict of special token names to IDs."""
    return {
        "PAD": PAD,
        "BOS": BOS,
        "EOS": EOS,
        "UNK": UNK,
        "PARSE_ERROR": PARSE_ERROR,
    }
