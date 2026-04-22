"""Byte-level tokenizer for raw source code.

Encodes Python source as raw UTF-8 bytes (0-255) with a PAD token at 256.
This preserves ALL surface syntax — whitespace, formatting, comments,
variable names — giving JEPA maximal nuisance to discard.

Used for the AST-vs-raw-code comparison experiment (Paper 3): if JEPA
on raw bytes doesn't collapse the same way as JEPA on AST tokens, then
AST tokenization (which strips nuisance) is the cause of collapse.

Vocabulary:
    0-255: Raw UTF-8 byte values
    256:   PAD token

Usage::

    tokens = byte_tokenize("def foo(): return 1", max_len=512)
    # [100, 101, 102, 32, 102, 111, 111, ..., 256, 256, ...]
"""
from __future__ import annotations

import numpy as np

PAD_TOKEN = 256
BYTE_VOCAB_SIZE = 257  # 0-255 bytes + PAD


def byte_tokenize(source: str, max_len: int = 512) -> np.ndarray:
    """Tokenize source code as raw UTF-8 bytes.

    Args:
        source: Python source code string.
        max_len: Fixed output length (padded or truncated).

    Returns:
        np.ndarray of shape (max_len,) with dtype uint16.
    """
    raw_bytes = source.encode("utf-8", errors="replace")
    tokens = np.full(max_len, PAD_TOKEN, dtype=np.uint16)
    n = min(len(raw_bytes), max_len)
    tokens[:n] = np.frombuffer(raw_bytes[:n], dtype=np.uint8)
    return tokens
