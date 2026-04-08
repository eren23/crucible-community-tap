#!/usr/bin/env python3
"""Rescore wandb runs with mean-of-last-N instead of peak-only.

Fetches metric history for every run in ``eren23/crucible-code-wm`` and
reports, per run:

- peak val_delta_cos (the number we used to report)
- mean ± std of val_delta_cos over the last N eval steps (default N=5)
- total steps run
- seed (from config, if present)

Output is a Markdown table written to stdout and, optionally, to ``--out``.
Used to replace the peak-only numbers in the Spider Chat final report with
honest mean ± std values.

Usage::

    python rescore_wandb_history.py \\
        --entity eren23 --project crucible-code-wm \\
        --metric val/delta_cos_sim --last-n 5 \\
        --out /tmp/wandb_rescore.md
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import Any

try:
    import wandb  # type: ignore
except ImportError:
    print("ERROR: wandb is not installed. pip install wandb", file=sys.stderr)
    sys.exit(1)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    n = len(values)
    mu = sum(values) / n
    if n == 1:
        return (mu, 0.0)
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return (mu, math.sqrt(var))


def rescore(
    entity: str,
    project: str,
    metric_key: str,
    last_n: int,
    name_filter: str | None = None,
) -> list[dict[str, Any]]:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=200)

    rows: list[dict[str, Any]] = []
    for run in runs:
        if name_filter and name_filter.lower() not in run.name.lower():
            continue
        if run.state not in {"finished", "running", "crashed"}:
            continue
        try:
            history = run.history(keys=[metric_key], pandas=False)
        except Exception as exc:
            print(f"  [warn] {run.name}: history fetch failed ({exc})", file=sys.stderr)
            continue

        values: list[float] = []
        for row in history:
            v = row.get(metric_key)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                continue
            values.append(fv)

        if not values:
            continue

        peak = max(values)
        mu, sigma = mean_std(values[-last_n:])
        cfg = run.config or {}
        seed = cfg.get("seed") or cfg.get("WM_SEED") or cfg.get("wm_seed")

        rows.append({
            "name": run.name,
            "id": run.id,
            "state": run.state,
            "steps": len(values),
            "peak": peak,
            "mean_last_n": mu,
            "std_last_n": sigma,
            "seed": seed,
        })

    rows.sort(key=lambda r: r["peak"], reverse=True)
    return rows


def format_markdown(rows: list[dict[str, Any]], metric_key: str, last_n: int) -> str:
    lines = [
        f"# Wandb rescore — `{metric_key}` (mean ± std over last {last_n} eval steps)",
        "",
        f"| Run | state | seed | evals | peak | mean-last-{last_n} | std-last-{last_n} |",
        f"|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        seed_s = str(r["seed"]) if r["seed"] is not None else "—"
        lines.append(
            f"| `{r['name']}` | {r['state']} | {seed_s} | {r['steps']} | "
            f"{r['peak']:.4f} | {r['mean_last_n']:.4f} | ±{r['std_last_n']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="eren23")
    parser.add_argument("--project", default="crucible-code-wm")
    parser.add_argument("--metric", default="val/delta_cos_sim")
    parser.add_argument("--last-n", type=int, default=5)
    parser.add_argument(
        "--filter",
        default=None,
        help="Substring to filter run names (case-insensitive). Defaults to all runs.",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = rescore(args.entity, args.project, args.metric, args.last_n, args.filter)
    md = format_markdown(rows, args.metric, args.last_n)

    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
