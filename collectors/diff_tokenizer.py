"""Diff tokenizer for code changes — tokenizes unified diffs into fixed-length sequences.

Extends the AST token vocabulary with diff-specific tokens for encoding
code changes (added/removed/modified lines). Designed for DeltaCodeWM
where we need to encode the CHANGE itself, not just before/after states.

Token layout (extends base AST vocab):
    0-661:    Base AST tokens (from ast_tokenizer.py)
    662:      DIFF_ADD     — marks an added line
    663:      DIFF_DEL     — marks a removed line
    664:      DIFF_CTX     — marks a context (unchanged) line
    665:      DIFF_HUNK    — marks a hunk separator (@@ ... @@)
    666:      DIFF_FILE    — marks a file header
    667:      DIFF_EMPTY   — empty diff (no change)
    668-699:  Reserved for future diff tokens

Usage:
    from collectors.diff_tokenizer import tokenize_diff, DIFF_VOCAB_SIZE

    diff_tokens = tokenize_diff(old_source, new_source, max_len=512)
"""
from __future__ import annotations

import difflib

import numpy as np

# Diff-specific tokens (appended after base AST vocab of 662)
DIFF_ADD = 662
DIFF_DEL = 663
DIFF_CTX = 664
DIFF_HUNK = 665
DIFF_FILE = 666
DIFF_EMPTY = 667

DIFF_VOCAB_SIZE = 700  # Base 662 + diff tokens + reserved

# Reuse base AST tokenizer for line content
PAD = 0
BOS = 1
EOS = 2


def _hash_token(s: str, n_buckets: int = 512, offset: int = 100) -> int:
    """FNV-1a hash of a string to a token bucket (same as AST tokenizer)."""
    h = 0x811C9DC5
    for ch in s.encode("utf-8", errors="replace"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return offset + (h % n_buckets)


def _tokenize_line(line: str, max_tokens: int = 15) -> list[int]:
    """Tokenize a single line of code into identifier-hash tokens.

    Simple word-level tokenization with FNV-1a hashing into 512 buckets.
    Keeps it consistent with the AST tokenizer's hash approach.
    """
    tokens = []
    # Split on whitespace and punctuation
    word = ""
    for ch in line:
        if ch.isalnum() or ch == "_":
            word += ch
        else:
            if word:
                tokens.append(_hash_token(word))
                word = ""
            if ch.strip():  # non-whitespace punctuation
                tokens.append(_hash_token(ch))
    if word:
        tokens.append(_hash_token(word))

    return tokens[:max_tokens]


def tokenize_diff(
    old_source: str,
    new_source: str,
    max_len: int = 512,
    context_lines: int = 1,
    max_tokens_per_line: int = 12,
) -> np.ndarray:
    """Tokenize a unified diff between two source code strings.

    Produces a fixed-length token sequence encoding ONLY the change:
    - Added lines get DIFF_ADD prefix + content tokens
    - Removed lines get DIFF_DEL prefix + content tokens
    - Context lines get DIFF_CTX prefix + content tokens (limited)
    - Hunk separators mark discontinuities

    Args:
        old_source: Before source code
        new_source: After source code
        max_len: Maximum token sequence length
        context_lines: Number of context lines around each change
        max_tokens_per_line: Max content tokens per diff line

    Returns:
        np.ndarray of shape (max_len,) with dtype int64
    """
    tokens = [BOS]

    # Handle edge cases
    if old_source == new_source:
        tokens.append(DIFF_EMPTY)
        tokens.append(EOS)
        return _pad(tokens, max_len)

    if not old_source.strip():
        # New file — all lines are additions
        tokens.append(DIFF_FILE)
        for line in new_source.splitlines()[:30]:
            tokens.append(DIFF_ADD)
            tokens.extend(_tokenize_line(line, max_tokens_per_line))
            if len(tokens) >= max_len - 2:
                break
        tokens.append(EOS)
        return _pad(tokens, max_len)

    if not new_source.strip():
        # Deleted file — all lines are removals
        tokens.append(DIFF_FILE)
        for line in old_source.splitlines()[:30]:
            tokens.append(DIFF_DEL)
            tokens.extend(_tokenize_line(line, max_tokens_per_line))
            if len(tokens) >= max_len - 2:
                break
        tokens.append(EOS)
        return _pad(tokens, max_len)

    # Compute unified diff
    old_lines = old_source.splitlines(keepends=True)
    new_lines = new_source.splitlines(keepends=True)

    diff_ops = list(difflib.unified_diff(
        old_lines, new_lines,
        n=context_lines,
        lineterm="",
    ))

    if not diff_ops:
        # No diff (whitespace-only change?)
        tokens.append(DIFF_EMPTY)
        tokens.append(EOS)
        return _pad(tokens, max_len)

    for line in diff_ops:
        if len(tokens) >= max_len - 2:
            break

        line_stripped = line.rstrip("\n\r")

        if line.startswith("@@"):
            tokens.append(DIFF_HUNK)
        elif line.startswith("---") or line.startswith("+++"):
            continue  # skip file headers (redundant with DIFF_FILE)
        elif line.startswith("+"):
            tokens.append(DIFF_ADD)
            tokens.extend(_tokenize_line(line_stripped[1:], max_tokens_per_line))
        elif line.startswith("-"):
            tokens.append(DIFF_DEL)
            tokens.extend(_tokenize_line(line_stripped[1:], max_tokens_per_line))
        else:
            tokens.append(DIFF_CTX)
            tokens.extend(_tokenize_line(line_stripped, max_tokens_per_line))

    tokens.append(EOS)
    return _pad(tokens, max_len)


def _pad(tokens: list[int], max_len: int) -> np.ndarray:
    """Pad or truncate to max_len."""
    if len(tokens) > max_len:
        tokens = tokens[:max_len - 1] + [EOS]
    else:
        tokens = tokens + [PAD] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.int64)


def diff_stats(old_source: str, new_source: str) -> dict:
    """Quick stats about a diff (for filtering/analysis)."""
    tokens = tokenize_diff(old_source, new_source, max_len=1024)
    n_add = int((tokens == DIFF_ADD).sum())
    n_del = int((tokens == DIFF_DEL).sum())
    n_ctx = int((tokens == DIFF_CTX).sum())
    n_hunk = int((tokens == DIFF_HUNK).sum())
    n_nonpad = int((tokens > 0).sum())
    is_empty = bool(DIFF_EMPTY in tokens)
    return {
        "n_add": n_add, "n_del": n_del, "n_ctx": n_ctx,
        "n_hunk": n_hunk, "n_nonpad": n_nonpad, "is_empty": is_empty,
    }
