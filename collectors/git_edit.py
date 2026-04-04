"""Git edit collector -- extracts edit sequences from git history.

Walks a repository's commit history and extracts (before_state, action, after_state)
transitions from file diffs. Each transition represents a localized code change:

- **before_tokens**: byte-level tokenized file content around the change region
- **edit_action**: (edit_type_onehot[3], line_offset_normalized) = 4-dim float vector
- **after_tokens**: byte-level tokenized file content after the change

Uses a simple byte-level tokenizer (bytes 0-255 + 4 special tokens) to stay
self-contained with no external tokenizer dependency.

HDF5 output schema::

    /metadata/repo_name        string
    /metadata/num_edits        int
    /metadata/vocab_size       int  (260)
    /metadata/context_window   int
    /edits/{idx}/before_tokens [context_window]  uint16
    /edits/{idx}/edit_action   [4]               float32
    /edits/{idx}/after_tokens  [context_window]  uint16

Usage::

    collector = GitEditCollector()
    stats = collector.collect(Path("/path/to/repo"), Path("edits.h5"), max_commits=500)
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseCollector, CollectionStats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Byte-level tokenizer
# ---------------------------------------------------------------------------

PAD = 256
UNK = 257
BOS = 258
EOS = 259
VOCAB_SIZE = 260


def _byte_tokenize(text: str, max_len: int) -> np.ndarray:
    """Encode *text* as byte-level token IDs, padded/truncated to *max_len*.

    Bytes 0-255 map to tokens 0-255.  BOS is prepended and EOS appended
    before padding with PAD.
    """
    raw = text.encode("utf-8", errors="replace")
    tokens = [BOS] + [b for b in raw] + [EOS]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens += [PAD] * (max_len - len(tokens))
    return np.array(tokens, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Edit types
# ---------------------------------------------------------------------------

EDIT_ADD = 0
EDIT_DELETE = 1
EDIT_MODIFY = 2

DEFAULT_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h",
})

# Regex to parse unified diff hunk headers: @@ -start[,count] +start[,count] @@
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


# ---------------------------------------------------------------------------
# Git helpers (subprocess, no gitpython)
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path, check: bool = False) -> tuple[str, int]:
    """Run a git command and return (stdout, returncode).

    Returns ("", -1) on timeout or missing git binary.
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.stdout, result.returncode
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "", -1


def _get_commit_hashes(repo: Path, max_commits: int) -> list[str]:
    """Return chronologically-ordered commit hashes (oldest first)."""
    out, _ = _git(["log", "--reverse", "--format=%H", f"-{max_commits}"], repo)
    return [h.strip() for h in out.splitlines() if h.strip()]


def _is_merge_commit(repo: Path, commit: str) -> bool:
    """Return True if *commit* has more than one parent."""
    _, rc = _git(["rev-parse", f"{commit}^2"], repo)
    # rev-parse succeeds (rc=0) only if ^2 (second parent) exists
    return rc == 0


def _get_changed_files(repo: Path, commit_a: str, commit_b: str) -> list[str]:
    """Return list of changed file paths between two commits."""
    out, _ = _git(["diff", "--name-only", commit_a, commit_b], repo)
    return [f.strip() for f in out.splitlines() if f.strip()]


def _get_file_content(repo: Path, commit: str, filepath: str) -> str | None:
    """Return file content at a given commit, or None on failure."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{filepath}"],
        cwd=str(repo),
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        # Binary file
        return None


def _get_diff_hunks(repo: Path, commit_a: str, commit_b: str, filepath: str) -> list[dict[str, Any]]:
    """Parse unified diff hunks for a single file between two commits.

    Returns list of dicts with keys: old_start, new_start, removed_lines, added_lines.
    """
    out, _ = _git(["diff", "-U0", commit_a, commit_b, "--", filepath], repo)
    if not out:
        return []

    hunks: list[dict[str, Any]] = []
    lines = out.splitlines()
    i = 0
    while i < len(lines):
        m = _HUNK_RE.match(lines[i])
        if m:
            old_start = int(m.group(1))
            new_start = int(m.group(2))
            removed: list[str] = []
            added: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("diff "):
                if lines[i].startswith("-"):
                    removed.append(lines[i][1:])
                elif lines[i].startswith("+"):
                    added.append(lines[i][1:])
                # context lines (space prefix) are skipped in -U0 mode
                i += 1
            hunks.append({
                "old_start": old_start,
                "new_start": new_start,
                "removed_lines": removed,
                "added_lines": added,
            })
        else:
            i += 1

    return hunks


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def _extract_context(content: str, line_num: int, context_window: int) -> str:
    """Extract text around *line_num* (1-based) that fits in *context_window* bytes.

    Tries to center the context around the change point.
    """
    if not content:
        return ""
    file_lines = content.splitlines(keepends=True)
    total = len(file_lines)
    if total == 0:
        return ""

    # Clamp to valid range
    target = max(0, min(line_num - 1, total - 1))

    # Expand outward from target until we fill the context window
    # (measured in bytes to match our tokenizer)
    start = target
    end = target + 1
    budget = context_window - 3  # reserve space for BOS, EOS, some margin

    while budget > 0 and (start > 0 or end < total):
        if start > 0:
            start -= 1
            budget -= len(file_lines[start].encode("utf-8", errors="replace"))
        if end < total and budget > 0:
            budget -= len(file_lines[end].encode("utf-8", errors="replace"))
            end += 1

    return "".join(file_lines[start:end])


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class GitEditCollector(BaseCollector):
    """Collect code edit transitions from git history."""

    def collect(
        self,
        source: Path,
        output: Path,
        *,
        max_commits: int = 1000,
        context_window: int = 256,
        file_extensions: list[str] | None = None,
        **kwargs: Any,
    ) -> CollectionStats:
        """Walk git log and extract edit transitions to HDF5.

        Parameters
        ----------
        source:
            Path to git repository.
        output:
            Path to output HDF5 file.
        max_commits:
            Maximum number of commits to process.
        context_window:
            Token context window size (before/after arrays).
        file_extensions:
            File extensions to include (default: common source code extensions).
        """
        import h5py

        extensions = frozenset(file_extensions) if file_extensions else DEFAULT_EXTENSIONS

        commits = _get_commit_hashes(source, max_commits)
        if len(commits) < 2:
            logger.warning("Repository has fewer than 2 commits -- nothing to collect")
            return CollectionStats(source=str(source), output_path=str(output))

        repo_name = source.resolve().name

        # Collect all transitions in memory first, then write HDF5
        transitions: list[dict[str, np.ndarray]] = []
        num_commits_processed = 0

        for i in range(len(commits) - 1):
            commit_a = commits[i]
            commit_b = commits[i + 1]

            # Skip merge commits
            if _is_merge_commit(source, commit_b):
                continue

            changed_files = _get_changed_files(source, commit_a, commit_b)
            if not changed_files:
                continue

            num_commits_processed += 1

            for filepath in changed_files:
                # Filter by extension
                ext = Path(filepath).suffix.lower()
                if ext not in extensions:
                    continue

                # Get file content before and after
                before_content = _get_file_content(source, commit_a, filepath)
                after_content = _get_file_content(source, commit_b, filepath)

                # Skip binary files or files that failed to decode
                if before_content is None and after_content is None:
                    continue

                # Determine edit type
                if before_content is None:
                    edit_type = EDIT_ADD
                    before_content = ""
                elif after_content is None:
                    edit_type = EDIT_DELETE
                    after_content = ""
                else:
                    edit_type = EDIT_MODIFY

                # Parse hunks
                hunks = _get_diff_hunks(source, commit_a, commit_b, filepath)
                if not hunks and edit_type == EDIT_MODIFY:
                    # No parseable hunks -- skip
                    continue

                # For ADD/DELETE with no hunks, create a single synthetic hunk
                if not hunks:
                    hunks = [{
                        "old_start": 1,
                        "new_start": 1,
                        "removed_lines": [],
                        "added_lines": [],
                    }]

                # One transition per hunk
                before_total_lines = len(before_content.splitlines()) if before_content else 1
                after_total_lines = len(after_content.splitlines()) if after_content else 1

                for hunk in hunks:
                    # Context around the change
                    before_ctx = _extract_context(
                        before_content, hunk["old_start"], context_window,
                    )
                    after_ctx = _extract_context(
                        after_content, hunk["new_start"], context_window,
                    )

                    before_tokens = _byte_tokenize(before_ctx, context_window)
                    after_tokens = _byte_tokenize(after_ctx, context_window)

                    # Action vector: edit_type one-hot (3) + normalized line offset (1)
                    action = np.zeros(4, dtype=np.float32)
                    action[edit_type] = 1.0
                    # Normalize line offset to [0, 1]
                    ref_lines = before_total_lines if edit_type != EDIT_ADD else after_total_lines
                    action[3] = hunk["old_start"] / max(ref_lines, 1)

                    transitions.append({
                        "before_tokens": before_tokens,
                        "edit_action": action,
                        "after_tokens": after_tokens,
                    })

            # Progress logging
            if num_commits_processed % 100 == 0:
                logger.info(
                    "Processed %d/%d commit pairs, %d transitions so far",
                    i + 1, len(commits) - 1, len(transitions),
                )

        # Write HDF5
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(output), "w") as f:
            # Metadata
            meta = f.create_group("metadata")
            meta.attrs["repo_name"] = repo_name
            meta.attrs["num_edits"] = len(transitions)
            meta.attrs["vocab_size"] = VOCAB_SIZE
            meta.attrs["context_window"] = context_window

            # Edits
            edits = f.create_group("edits")
            for idx, t in enumerate(transitions):
                g = edits.create_group(str(idx))
                g.create_dataset("before_tokens", data=t["before_tokens"])
                g.create_dataset("edit_action", data=t["edit_action"])
                g.create_dataset("after_tokens", data=t["after_tokens"])

        logger.info(
            "Written %d transitions from %d commits to %s",
            len(transitions), num_commits_processed, output,
        )

        return CollectionStats(
            num_transitions=len(transitions),
            num_sequences=num_commits_processed,
            source=str(source),
            output_path=str(output),
            metadata={
                "repo_name": repo_name,
                "vocab_size": VOCAB_SIZE,
                "context_window": context_window,
                "max_commits": max_commits,
                "extensions": sorted(extensions),
            },
        )

    def validate(self, output: Path) -> bool:
        """Verify HDF5 integrity and expected schema."""
        import h5py

        try:
            with h5py.File(str(output), "r") as f:
                # Check metadata group
                if "metadata" not in f:
                    logger.error("Missing /metadata group")
                    return False
                meta = f["metadata"]
                for key in ("repo_name", "num_edits", "vocab_size", "context_window"):
                    if key not in meta.attrs:
                        logger.error("Missing metadata attribute: %s", key)
                        return False

                num_edits = int(meta.attrs["num_edits"])
                context_window = int(meta.attrs["context_window"])

                # Check edits group
                if "edits" not in f:
                    logger.error("Missing /edits group")
                    return False

                edits = f["edits"]
                if len(edits) != num_edits:
                    logger.error(
                        "Edit count mismatch: metadata says %d, found %d groups",
                        num_edits, len(edits),
                    )
                    return False

                # Spot-check first and last edit
                for idx_str in [str(0), str(num_edits - 1)] if num_edits > 0 else []:
                    if idx_str not in edits:
                        logger.error("Missing edit group: %s", idx_str)
                        return False
                    g = edits[idx_str]
                    for ds_name, expected_shape in [
                        ("before_tokens", (context_window,)),
                        ("edit_action", (4,)),
                        ("after_tokens", (context_window,)),
                    ]:
                        if ds_name not in g:
                            logger.error("Missing dataset %s in edit %s", ds_name, idx_str)
                            return False
                        if g[ds_name].shape != expected_shape:
                            logger.error(
                                "Shape mismatch for %s/%s: expected %s, got %s",
                                idx_str, ds_name, expected_shape, g[ds_name].shape,
                            )
                            return False

            return True
        except Exception:
            logger.exception("Validation failed for %s", output)
            return False
