"""Trajectory collector -- chains CommitPack edits into per-file edit sequences.

Groups CommitPack records by (repo, file_path), then chains consecutive edits
via content-hash matching: if record A's new_contents hashes to the same value
as record B's old_contents, B follows A.  Longest chains become trajectories.

Also supports extracting perfect chronological trajectories from local git repos
(cloned or local), using ``git log`` for ordering.

HDF5 output schema (backwards-compatible + trajectory index)::

    # Flat arrays -- existing CodeStateAdapter reads these unchanged
    /before_tokens   [total_transitions, context_window]  uint16
    /after_tokens    [total_transitions, context_window]  uint16
    /edit_actions    [total_transitions, action_dim]       float32

    # Trajectory index -- new, ignored by old code
    /trajectory/traj_id       [total_transitions]  int32
    /trajectory/step_in_traj  [total_transitions]  int32
    /trajectory/traj_offsets  [num_trajectories]   int64
    /trajectory/traj_lengths  [num_trajectories]   int32

    # Metadata (extended)
    /metadata  attrs: ..., has_trajectories=True, num_trajectories, max_traj_len,
                      mean_traj_len, source_repos (for git mode)

Usage::

    # From CommitPack (chain reconstruction)
    collector = TrajectoryCollector()
    stats = collector.collect_from_commitpack(
        Path("trajectories.h5"), max_samples=500_000, min_traj_len=4,
    )

    # From local git repos
    stats = collector.collect_from_git(
        [Path("/path/to/repo1"), Path("/path/to/repo2")],
        Path("git_trajectories.h5"), min_edits=4,
    )
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .ast_tokenizer import ast_tokenize, get_vocab_size
from .ast_diff import compute_rich_action, ACTION_DIM_RICH
from .commitpack_processor import compute_action, ACTION_DIM
from .base import BaseCollector, CollectionStats

# Batch size for reading HDF5 chunks (memory vs speed tradeoff)
_HDF5_READ_CHUNK = 50_000

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """Fast content hash for chaining. Uses md5 (not crypto, just matching)."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# CommitPack trajectory chaining
# ---------------------------------------------------------------------------

def _chain_records(
    records: list[dict[str, str]],
) -> list[list[dict[str, str]]]:
    """Chain records where one's new_contents matches another's old_contents.

    Returns list of chains (longest first), each chain being a list of records
    in order.  Each record is used at most once.
    """
    if len(records) < 2:
        return []

    # Build hash index: hash(old_contents) -> list of record indices
    old_hash_idx: dict[str, list[int]] = defaultdict(list)
    new_hashes: list[str] = []

    for i, rec in enumerate(records):
        oh = _content_hash(rec["old"])
        nh = _content_hash(rec["new"])
        old_hash_idx[oh].append(i)
        new_hashes.append(nh)

    # Build adjacency: record i -> record j if hash(i.new) == hash(j.old)
    # This means j's old_contents matches i's new_contents (j follows i)
    adjacency: dict[int, list[int]] = defaultdict(list)
    has_predecessor: set[int] = set()

    for i, nh in enumerate(new_hashes):
        for j in old_hash_idx.get(nh, []):
            if j != i:
                adjacency[i].append(j)
                has_predecessor.add(j)

    # Find chain starts (records with no predecessor)
    starts = [i for i in range(len(records)) if i not in has_predecessor]

    # Walk chains greedily from each start
    used: set[int] = set()
    chains: list[list[dict[str, str]]] = []

    for start in starts:
        chain_indices = [start]
        used.add(start)
        current = start

        while True:
            nexts = [j for j in adjacency.get(current, []) if j not in used]
            if not nexts:
                break
            # Pick first available successor
            nxt = nexts[0]
            chain_indices.append(nxt)
            used.add(nxt)
            current = nxt

        if len(chain_indices) >= 2:
            chains.append([records[i] for i in chain_indices])

    # Sort by length descending
    chains.sort(key=len, reverse=True)
    return chains


# ---------------------------------------------------------------------------
# Chain from existing HDF5 (no re-download)
# ---------------------------------------------------------------------------

def _row_hash(row: np.ndarray) -> int:
    """Fast hash for a token row. Uses tobytes() for speed."""
    return hash(row.tobytes())


def chain_from_hdf5(
    hdf5_path: Path,
    *,
    min_traj_len: int = 4,
    max_traj_len: int = 20,
    max_bucket_size: int = 50,
    progress_interval: int = 100_000,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Discover trajectory chains in an existing flat HDF5 by token-row hashing.

    Scans before_tokens and after_tokens, builds hash index, chains rows where
    after_tokens[i] == before_tokens[j] (exact match on token sequences).

    Buckets with more than ``max_bucket_size`` entries are skipped — these are
    generic AST patterns (empty files, simple __init__.py) that don't form
    meaningful trajectories and cause quadratic blowup.

    Returns (chains, info) where:
        - chains: list of index lists, each chain = [row_i, row_j, row_k, ...]
          meaning transition i -> j -> k (after[i]==before[j], after[j]==before[k])
        - info: dict with statistics

    This is O(N) in memory (hash table) and O(N) in time (with bucket cap).
    """
    import h5py

    f = h5py.File(str(hdf5_path), "r")
    n = f["before_tokens"].shape[0]

    logger.info("Scanning %d rows for trajectory chains (bucket cap=%d)...", n, max_bucket_size)

    # Phase 1: Hash all before_tokens rows -> index
    before_hash_to_rows: dict[int, list[int]] = defaultdict(list)

    for start in range(0, n, _HDF5_READ_CHUNK):
        end = min(start + _HDF5_READ_CHUNK, n)
        chunk = f["before_tokens"][start:end]
        for local_i in range(chunk.shape[0]):
            h = _row_hash(chunk[local_i])
            before_hash_to_rows[h].append(start + local_i)
        if end % progress_interval < _HDF5_READ_CHUNK:
            logger.info("  Hashed before_tokens: %d/%d rows", end, n)

    # Prune mega-buckets (generic patterns that cause O(N^2) blowup)
    n_pruned = 0
    pruned_rows = 0
    for h in list(before_hash_to_rows):
        if len(before_hash_to_rows[h]) > max_bucket_size:
            pruned_rows += len(before_hash_to_rows[h])
            del before_hash_to_rows[h]
            n_pruned += 1

    logger.info(
        "  before_tokens index: %d unique hashes from %d rows "
        "(pruned %d mega-buckets covering %d rows)",
        len(before_hash_to_rows), n, n_pruned, pruned_rows,
    )

    # Phase 2: For each after_tokens row, find matching before_tokens rows
    adjacency: dict[int, list[int]] = defaultdict(list)
    has_predecessor: set[int] = set()
    n_edges = 0

    for start in range(0, n, _HDF5_READ_CHUNK):
        end = min(start + _HDF5_READ_CHUNK, n)
        after_chunk = f["after_tokens"][start:end]

        for local_i in range(after_chunk.shape[0]):
            global_i = start + local_i
            h = _row_hash(after_chunk[local_i])
            candidates = before_hash_to_rows.get(h, [])
            for j in candidates:
                if j != global_i:
                    adjacency[global_i].append(j)
                    has_predecessor.add(j)
                    n_edges += 1

        if end % progress_interval < _HDF5_READ_CHUNK:
            logger.info("  Matched after_tokens: %d/%d rows (%d edges)", end, n, n_edges)

    f.close()
    logger.info("  Adjacency: %d edges from %d source rows", n_edges, len(adjacency))

    # Phase 3: Walk chains from roots (rows with no predecessor)
    roots = [i for i in range(n) if i not in has_predecessor and i in adjacency]
    logger.info("  Found %d chain roots, %d rows with successors", len(roots), len(adjacency))

    used: set[int] = set()
    chains: list[list[int]] = []

    for root in roots:
        chain = [root]
        used.add(root)
        current = root

        while len(chain) <= max_traj_len:
            nexts = [j for j in adjacency.get(current, []) if j not in used]
            if not nexts:
                break
            nxt = nexts[0]
            chain.append(nxt)
            used.add(nxt)
            current = nxt

        # Chain of length L has L transitions (L+1 states via before/after)
        # But since each row IS a transition, chain of L rows = L transitions
        if len(chain) >= min_traj_len:
            chains.append(chain)

    chains.sort(key=len, reverse=True)

    total_transitions = sum(len(c) for c in chains)
    info = {
        "num_chains": len(chains),
        "total_transitions_in_chains": total_transitions,
        "total_rows_scanned": n,
        "chain_lengths": [len(c) for c in chains[:20]],  # first 20 for display
        "mean_chain_len": float(np.mean([len(c) for c in chains])) if chains else 0,
        "max_chain_len": max(len(c) for c in chains) if chains else 0,
        "coverage_pct": 100.0 * total_transitions / max(n, 1),
    }

    logger.info(
        "Chaining complete: %d chains (%d transitions, %.1f%% coverage), "
        "mean len %.1f, max len %d",
        info["num_chains"], total_transitions, info["coverage_pct"],
        info["mean_chain_len"], info["max_chain_len"],
    )

    return chains, info


# ---------------------------------------------------------------------------
# Git trajectory extraction
# ---------------------------------------------------------------------------

def _run_git(repo: Path, *args: str, timeout: int = 30) -> str:
    """Run a git command, return stdout or empty string on failure."""
    try:
        res = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return res.stdout if res.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _file_at_rev(repo: Path, rev: str, path: str) -> str:
    """Get file content at a specific git revision."""
    return _run_git(repo, "show", f"{rev}:{path}")


def collect_git_trajectories(
    repo: Path,
    *,
    min_edits: int = 4,
    max_commits: int = 5000,
    max_trajs: int = 500,
    max_file_size: int = 50_000,
    extensions: frozenset[str] = frozenset({".py"}),
) -> list[dict[str, Any]]:
    """Extract per-file edit trajectories from a git repository.

    Returns list of trajectory dicts, each containing:
        - file: relative file path
        - shas: list of commit hashes (chronological)
        - messages: list of commit messages
        - states: list of file contents at each commit
    """
    log = _run_git(
        repo, "log", "--no-merges", f"-n{max_commits}",
        "--pretty=format:%H%x00%s", "--name-only",
        timeout=60,
    )
    if not log:
        return []

    # Group (sha, message) by file path
    file_shas: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for block in log.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if not lines or "\x00" not in lines[0]:
            continue
        sha, subject = lines[0].split("\x00", 1)
        for f in lines[1:]:
            f = f.strip()
            if f and any(f.endswith(ext) for ext in extensions):
                file_shas[f].append((sha, subject))

    trajectories = []

    for py_file, history in file_shas.items():
        # Need min_edits + 1 commits to get min_edits transitions
        if len(history) < min_edits + 1:
            continue

        # Reverse for chronological order (git log is newest-first)
        chain = list(reversed(history[: min_edits + 1]))
        shas = [c[0] for c in chain]

        # Fetch file contents at each commit
        states: list[str] = []
        skip = False
        for sha in shas:
            content = _file_at_rev(repo, sha, py_file)
            if not content.strip() or len(content) > max_file_size:
                skip = True
                break
            states.append(content)

        if skip or len(states) != min_edits + 1:
            continue

        # Skip if all states are identical (no real changes)
        if len(set(states)) == 1:
            continue

        trajectories.append({
            "file": py_file,
            "shas": shas,
            "messages": [c[1][:100] for c in chain],
            "states": states,
        })

        if len(trajectories) >= max_trajs:
            break

    return trajectories


# ---------------------------------------------------------------------------
# HDF5 writer with trajectory index
# ---------------------------------------------------------------------------

def _write_trajectory_hdf5(
    output: Path,
    all_before: list[np.ndarray],
    all_after: list[np.ndarray],
    all_actions: list[np.ndarray],
    traj_ids: list[int],
    steps_in_traj: list[int],
    traj_lengths: list[int],
    context_window: int,
    action_dim: int,
    source_name: str,
    source_repos: list[str] | None = None,
) -> None:
    """Write trajectory-structured HDF5, backwards-compatible with flat schema."""
    import h5py

    n = len(all_before)
    n_traj = len(traj_lengths)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output), "w") as f:
        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["vocab_size"] = get_vocab_size()
        meta.attrs["context_window"] = context_window
        meta.attrs["action_dim"] = action_dim
        meta.attrs["num_edits"] = n
        meta.attrs["tokenizer"] = "python_ast"
        meta.attrs["source"] = source_name
        meta.attrs["has_trajectories"] = True
        meta.attrs["num_trajectories"] = n_traj
        meta.attrs["max_traj_len"] = max(traj_lengths) if traj_lengths else 0
        meta.attrs["mean_traj_len"] = float(np.mean(traj_lengths)) if traj_lengths else 0.0
        if source_repos:
            meta.attrs["source_repos"] = ",".join(source_repos[:50])

        # Flat arrays (backwards-compatible)
        chunk_rows = min(1024, max(n, 1))
        f.create_dataset(
            "before_tokens",
            data=np.stack(all_before) if all_before else np.empty((0, context_window), dtype=np.uint16),
            dtype="uint16",
            chunks=(chunk_rows, context_window) if n > 0 else None,
        )
        f.create_dataset(
            "after_tokens",
            data=np.stack(all_after) if all_after else np.empty((0, context_window), dtype=np.uint16),
            dtype="uint16",
            chunks=(chunk_rows, context_window) if n > 0 else None,
        )
        f.create_dataset(
            "edit_actions",
            data=np.stack(all_actions) if all_actions else np.empty((0, action_dim), dtype=np.float32),
            dtype="float32",
            chunks=(chunk_rows, action_dim) if n > 0 else None,
        )

        # Trajectory index
        traj_grp = f.create_group("trajectory")
        traj_grp.create_dataset("traj_id", data=np.array(traj_ids, dtype=np.int32))
        traj_grp.create_dataset("step_in_traj", data=np.array(steps_in_traj, dtype=np.int32))

        # Compute offsets from lengths
        offsets = np.zeros(n_traj, dtype=np.int64)
        if n_traj > 1:
            offsets[1:] = np.cumsum(traj_lengths[:-1])
        traj_grp.create_dataset("traj_offsets", data=offsets)
        traj_grp.create_dataset("traj_lengths", data=np.array(traj_lengths, dtype=np.int32))

    logger.info(
        "Written trajectory HDF5: %d transitions in %d trajectories "
        "(mean len %.1f, max len %d) -> %s",
        n, n_traj,
        float(np.mean(traj_lengths)) if traj_lengths else 0,
        max(traj_lengths) if traj_lengths else 0,
        output,
    )


# ---------------------------------------------------------------------------
# Tokenize + action for a trajectory
# ---------------------------------------------------------------------------

def _process_trajectory_states(
    states: list[str],
    context_window: int,
    rich_actions: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
    """Tokenize states and compute actions for a single trajectory.

    Returns (before_list, after_list, action_list) for len(states)-1 transitions,
    or None if any state fails to tokenize.
    """
    before_list: list[np.ndarray] = []
    after_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []

    for i in range(len(states) - 1):
        try:
            before_tokens = ast_tokenize(states[i], context_window)
            after_tokens = ast_tokenize(states[i + 1], context_window)

            if rich_actions:
                action = compute_rich_action(states[i], states[i + 1])
            else:
                action = compute_action(states[i], states[i + 1])

            before_list.append(before_tokens)
            after_list.append(after_tokens)
            action_list.append(action)
        except Exception as e:
            logger.debug("Skipping transition %d in trajectory: %s", i, e)
            return None

    return before_list, after_list, action_list


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class TrajectoryCollector(BaseCollector):
    """Collect trajectory-structured code edit data.

    Two modes:
    - ``collect()``: Chain CommitPack records into per-file trajectories
    - ``collect_from_git()``: Extract perfect trajectories from git repos
    """

    def collect(
        self,
        source: Path,
        output: Path,
        *,
        max_samples: int | None = None,
        use_ft: bool = True,
        context_window: int = 512,
        max_file_size: int = 50_000,
        rich_actions: bool = False,
        min_traj_len: int = 4,
        max_traj_len: int = 20,
        **kwargs: Any,
    ) -> CollectionStats:
        """Chain CommitPack records into per-file edit trajectories.

        Parameters
        ----------
        source:
            Ignored (data comes from HuggingFace).
        output:
            Output HDF5 path.
        max_samples:
            Max CommitPack records to scan. None = all.
        use_ft:
            True for CommitPackFT (56K), False for full CommitPack.
        context_window:
            AST token sequence length.
        max_file_size:
            Skip files larger than this.
        rich_actions:
            Use 15-dim actions (default: 7-dim).
        min_traj_len:
            Minimum trajectory length in transitions (states - 1).
        max_traj_len:
            Maximum trajectory length (truncate longer chains).
        """
        dataset_name = "bigcode/commitpackft" if use_ft else "bigcode/commitpack"
        logger.info(
            "Loading %s for trajectory extraction (max_samples=%s, min_traj=%d)",
            dataset_name, max_samples, min_traj_len,
        )

        # --- Phase 1: Load records, group by file identity ---
        file_groups = self._load_and_group(
            dataset_name, use_ft, max_samples, max_file_size,
        )

        # --- Phase 2: Chain records within each group ---
        trajectories = self._chain_groups(file_groups, min_traj_len, max_traj_len)

        if not trajectories:
            logger.warning("No trajectories found -- nothing to write")
            return CollectionStats(
                source=dataset_name,
                output_path=str(output),
                metadata={"error": "no trajectories found"},
            )

        # --- Phase 3: Tokenize and write ---
        action_dim = ACTION_DIM_RICH if rich_actions else ACTION_DIM
        stats = self._tokenize_and_write(
            trajectories, output, context_window, rich_actions,
            action_dim, dataset_name,
        )
        return stats

    def collect_from_git(
        self,
        repos: list[Path],
        output: Path,
        *,
        context_window: int = 512,
        rich_actions: bool = False,
        min_edits: int = 4,
        max_commits: int = 5000,
        max_trajs_per_repo: int = 500,
        **kwargs: Any,
    ) -> CollectionStats:
        """Extract per-file trajectories from local git repositories.

        Parameters
        ----------
        repos:
            List of paths to git repositories.
        output:
            Output HDF5 path.
        context_window:
            AST token sequence length.
        rich_actions:
            Use 15-dim actions (default: 7-dim).
        min_edits:
            Minimum transitions per trajectory.
        max_commits:
            Max commits to scan per repo.
        max_trajs_per_repo:
            Max trajectories per repo.
        """
        all_trajectories: list[list[str]] = []
        source_repos: list[str] = []

        for repo in repos:
            repo = repo.resolve()
            if not (repo / ".git").exists():
                logger.warning("Not a git repo, skipping: %s", repo)
                continue

            logger.info("Extracting trajectories from %s...", repo.name)
            trajs = collect_git_trajectories(
                repo,
                min_edits=min_edits,
                max_commits=max_commits,
                max_trajs=max_trajs_per_repo,
            )

            for t in trajs:
                all_trajectories.append(t["states"])
            source_repos.append(repo.name)

            logger.info(
                "  %s: %d trajectories (min %d transitions)",
                repo.name, len(trajs), min_edits,
            )

        if not all_trajectories:
            logger.warning("No trajectories extracted from any repo")
            return CollectionStats(
                source="git",
                output_path=str(output),
                metadata={"error": "no trajectories"},
            )

        action_dim = ACTION_DIM_RICH if rich_actions else ACTION_DIM
        stats = self._tokenize_and_write(
            all_trajectories, output, context_window, rich_actions,
            action_dim, f"git:{','.join(source_repos[:10])}",
            source_repos=source_repos,
        )
        return stats

    def collect_from_hdf5(
        self,
        source_hdf5: Path,
        output: Path,
        *,
        min_traj_len: int = 4,
        max_traj_len: int = 20,
        **kwargs: Any,
    ) -> CollectionStats:
        """Discover and extract trajectories from an existing flat HDF5.

        Chains rows where after_tokens[i] == before_tokens[j] into trajectories.
        Copies the chained transitions into a new HDF5 with trajectory index.
        No re-download, no re-tokenization — just reads existing arrays.

        Parameters
        ----------
        source_hdf5:
            Existing flat HDF5 (e.g. commitpack_python_1.5m.h5).
        output:
            Output HDF5 with trajectory structure.
        min_traj_len:
            Minimum transitions per chain.
        max_traj_len:
            Maximum transitions per chain (truncate longer).
        """
        import h5py

        # Phase 1: Discover chains
        chains, info = chain_from_hdf5(
            source_hdf5,
            min_traj_len=min_traj_len,
            max_traj_len=max_traj_len,
        )

        if not chains:
            logger.warning("No chains found in %s", source_hdf5)
            return CollectionStats(
                source=str(source_hdf5),
                output_path=str(output),
                metadata={"error": "no chains found", **info},
            )

        # Phase 2: Copy chained rows to new HDF5
        logger.info("Copying %d chains to %s...", len(chains), output)

        f_src = h5py.File(str(source_hdf5), "r")
        meta_src = f_src["metadata"]
        context_window = int(meta_src.attrs["context_window"])
        action_dim = int(meta_src.attrs["action_dim"])
        vocab_size = int(meta_src.attrs["vocab_size"])
        source_name = str(meta_src.attrs.get("source", str(source_hdf5)))

        all_before: list[np.ndarray] = []
        all_after: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        traj_ids: list[int] = []
        steps_in_traj: list[int] = []
        traj_lengths: list[int] = []

        for traj_idx, chain in enumerate(chains):
            # Each row in the chain is a transition index in the source HDF5
            # Read them in bulk for efficiency
            indices = sorted(set(chain))
            idx_list = [int(i) for i in indices]

            before_batch = f_src["before_tokens"][idx_list]
            after_batch = f_src["after_tokens"][idx_list]
            action_batch = f_src["edit_actions"][idx_list]

            # Map sorted unique indices back to chain order
            idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}

            for step, row_idx in enumerate(chain):
                pos = idx_to_pos[row_idx]
                all_before.append(before_batch[pos])
                all_after.append(after_batch[pos])
                all_actions.append(action_batch[pos])
                traj_ids.append(traj_idx)
                steps_in_traj.append(step)

            traj_lengths.append(len(chain))

            if (traj_idx + 1) % 500 == 0:
                logger.info("  Copied %d/%d chains (%d transitions)",
                            traj_idx + 1, len(chains), len(all_before))

        f_src.close()

        # Phase 3: Write output
        _write_trajectory_hdf5(
            output=output,
            all_before=all_before,
            all_after=all_after,
            all_actions=all_actions,
            traj_ids=traj_ids,
            steps_in_traj=steps_in_traj,
            traj_lengths=traj_lengths,
            context_window=context_window,
            action_dim=action_dim,
            source_name=f"chained:{source_name}",
        )

        return CollectionStats(
            num_transitions=len(all_before),
            num_sequences=len(traj_lengths),
            source=str(source_hdf5),
            output_path=str(output),
            metadata={
                "num_trajectories": len(traj_lengths),
                "total_transitions": len(all_before),
                "mean_traj_len": float(np.mean(traj_lengths)),
                "max_traj_len": max(traj_lengths),
                "min_traj_len": min(traj_lengths),
                "vocab_size": vocab_size,
                "context_window": context_window,
                "action_dim": action_dim,
                "source_rows_scanned": info["total_rows_scanned"],
                "coverage_pct": info["coverage_pct"],
            },
        )

    # --- Internal methods ---

    def _load_and_group(
        self,
        dataset_name: str,
        use_ft: bool,
        max_samples: int | None,
        max_file_size: int,
    ) -> dict[str, list[dict[str, str]]]:
        """Load CommitPack records and group by (repo, file_path)."""
        from .commitpack_processor import CommitPackProcessor

        # Reuse the shard iterator from CommitPackProcessor
        processor = CommitPackProcessor()

        if not use_ft:
            ds = processor._iter_commitpack_shards(
                dataset_name, max_samples=max_samples,
            )
        else:
            from datasets import load_dataset
            jsonl_url = f"hf://datasets/{dataset_name}/data/python/data.jsonl"
            try:
                ds = load_dataset(
                    "json", data_files=jsonl_url, split="train", streaming=True,
                )
            except Exception:
                ds = load_dataset(
                    dataset_name, "python", split="train", streaming=True,
                )

        # Group records by file identity
        file_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        processed = 0
        skipped = 0

        for sample in ds:
            if max_samples is not None and processed >= max_samples:
                break

            old_contents: str = sample.get("old_contents", "") or ""
            new_contents: str = sample.get("new_contents", "") or ""

            # Skip empty or oversized
            if not old_contents.strip() and not new_contents.strip():
                skipped += 1
                continue
            if len(old_contents) > max_file_size or len(new_contents) > max_file_size:
                skipped += 1
                continue

            # File identity key: use new_file (or old_file) + first repo
            new_file = sample.get("new_file", sample.get("old_file", ""))
            repos = sample.get("repos", "")
            if isinstance(repos, list):
                repo_key = repos[0] if repos else "unknown"
            else:
                repo_key = repos.split(",")[0].strip() if repos else "unknown"

            file_key = f"{repo_key}:{new_file}"

            file_groups[file_key].append({
                "old": old_contents,
                "new": new_contents,
            })

            processed += 1
            if processed % 50_000 == 0:
                n_groups = len(file_groups)
                logger.info(
                    "Scanned %d records -> %d file groups (skipped %d)",
                    processed, n_groups, skipped,
                )

        logger.info(
            "Scan complete: %d records -> %d file groups (skipped %d)",
            processed, len(file_groups), skipped,
        )
        return file_groups

    def _chain_groups(
        self,
        file_groups: dict[str, list[dict[str, str]]],
        min_traj_len: int,
        max_traj_len: int,
    ) -> list[list[str]]:
        """Chain records within each file group into trajectories.

        Returns list of trajectories, each a list of file content strings
        (states). A trajectory of N states yields N-1 transitions.
        """
        trajectories: list[list[str]] = []
        groups_with_chains = 0

        for file_key, records in file_groups.items():
            if len(records) < 2:
                continue

            chains = _chain_records(records)

            for chain in chains:
                if len(chain) < min_traj_len + 1:
                    continue

                # Build state sequence: [old_0, new_0=old_1, new_1=old_2, ..., new_n]
                # The chain is already ordered so that chain[i].new == chain[i+1].old
                states = [chain[0]["old"]]
                for rec in chain:
                    states.append(rec["new"])

                # Truncate if needed
                if len(states) > max_traj_len + 1:
                    states = states[: max_traj_len + 1]

                # Deduplicate consecutive identical states
                deduped = [states[0]]
                for s in states[1:]:
                    if s != deduped[-1]:
                        deduped.append(s)

                if len(deduped) >= min_traj_len + 1:
                    trajectories.append(deduped)
                    groups_with_chains += 1

        logger.info(
            "Chaining complete: %d trajectories from %d file groups "
            "(mean len %.1f states)",
            len(trajectories), groups_with_chains,
            float(np.mean([len(t) for t in trajectories])) if trajectories else 0,
        )
        return trajectories

    def _tokenize_and_write(
        self,
        trajectories: list[list[str]],
        output: Path,
        context_window: int,
        rich_actions: bool,
        action_dim: int,
        source_name: str,
        source_repos: list[str] | None = None,
    ) -> CollectionStats:
        """Tokenize trajectories and write HDF5."""
        all_before: list[np.ndarray] = []
        all_after: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        traj_ids: list[int] = []
        steps_in_traj: list[int] = []
        traj_lengths: list[int] = []

        valid_traj_id = 0

        for traj_idx, states in enumerate(trajectories):
            result = _process_trajectory_states(states, context_window, rich_actions)
            if result is None:
                continue

            before_list, after_list, action_list = result
            n_transitions = len(before_list)

            all_before.extend(before_list)
            all_after.extend(after_list)
            all_actions.extend(action_list)

            for step in range(n_transitions):
                traj_ids.append(valid_traj_id)
                steps_in_traj.append(step)

            traj_lengths.append(n_transitions)
            valid_traj_id += 1

            if (traj_idx + 1) % 100 == 0:
                logger.info(
                    "Tokenized %d/%d trajectories (%d transitions so far)",
                    traj_idx + 1, len(trajectories), len(all_before),
                )

        if not all_before:
            logger.warning("No valid transitions after tokenization")
            return CollectionStats(
                source=source_name,
                output_path=str(output),
                metadata={"error": "no valid transitions"},
            )

        _write_trajectory_hdf5(
            output=output,
            all_before=all_before,
            all_after=all_after,
            all_actions=all_actions,
            traj_ids=traj_ids,
            steps_in_traj=steps_in_traj,
            traj_lengths=traj_lengths,
            context_window=context_window,
            action_dim=action_dim,
            source_name=source_name,
            source_repos=source_repos,
        )

        return CollectionStats(
            num_transitions=len(all_before),
            num_sequences=len(traj_lengths),
            source=source_name,
            output_path=str(output),
            metadata={
                "num_trajectories": len(traj_lengths),
                "total_transitions": len(all_before),
                "mean_traj_len": float(np.mean(traj_lengths)),
                "max_traj_len": max(traj_lengths),
                "min_traj_len": min(traj_lengths),
                "vocab_size": get_vocab_size(),
                "context_window": context_window,
                "action_dim": action_dim,
                "rich_actions": rich_actions,
            },
        )

    def validate(self, output: Path) -> bool:
        """Verify trajectory HDF5 integrity."""
        import h5py

        try:
            with h5py.File(str(output), "r") as f:
                if "metadata" not in f:
                    logger.error("Missing /metadata group")
                    return False

                meta = f["metadata"]
                for key in ("num_edits", "context_window", "action_dim", "vocab_size"):
                    if key not in meta.attrs:
                        logger.error("Missing metadata attribute: %s", key)
                        return False

                n = int(meta.attrs["num_edits"])
                cw = int(meta.attrs["context_window"])
                ad = int(meta.attrs["action_dim"])

                # Check flat arrays
                for ds_name, expected_shape in [
                    ("before_tokens", (n, cw)),
                    ("after_tokens", (n, cw)),
                    ("edit_actions", (n, ad)),
                ]:
                    if ds_name not in f:
                        logger.error("Missing dataset: %s", ds_name)
                        return False
                    if f[ds_name].shape != expected_shape:
                        logger.error(
                            "Shape mismatch for %s: expected %s, got %s",
                            ds_name, expected_shape, f[ds_name].shape,
                        )
                        return False

                # Check trajectory index
                if not meta.attrs.get("has_trajectories", False):
                    logger.error("Missing has_trajectories flag")
                    return False

                if "trajectory" not in f:
                    logger.error("Missing /trajectory group")
                    return False

                traj = f["trajectory"]
                for ds_name in ("traj_id", "step_in_traj", "traj_offsets", "traj_lengths"):
                    if ds_name not in traj:
                        logger.error("Missing trajectory dataset: %s", ds_name)
                        return False

                # Consistency checks
                if traj["traj_id"].shape[0] != n:
                    logger.error("traj_id length != num_edits")
                    return False

                n_traj = int(meta.attrs["num_trajectories"])
                if traj["traj_lengths"].shape[0] != n_traj:
                    logger.error("traj_lengths length != num_trajectories")
                    return False

                # Verify offsets + lengths sum to n
                lengths = traj["traj_lengths"][:]
                if int(lengths.sum()) != n:
                    logger.error(
                        "Sum of traj_lengths (%d) != num_edits (%d)",
                        int(lengths.sum()), n,
                    )
                    return False

            return True
        except Exception:
            logger.exception("Validation failed for %s", output)
            return False
