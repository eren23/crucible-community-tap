"""Zero-shot Diff-XYZ eval runner — local HF model, no SFT.

Loads a base HF model in 4-bit and runs the Diff-XYZ harness against it.
The point: anchor our pipeline against the paper's no-SFT baselines (e.g.
Table 5: Qwen2.5-Coder-7B Diff-Gen search-replace EM=0.68). Without that
anchor we can't tell whether a fine-tuned-then-eval'd EM gap is "the SFT
recipe is wrong" or "our eval pipeline differs from the paper".

Env vars (all optional with sane defaults):
    DIFFXYZ_BASE_MODEL      HF model id, default 'Qwen/Qwen2.5-Coder-7B-Instruct'
    DIFFXYZ_TASK            apply | anti_apply | diff_gen, default 'diff_gen'
    DIFFXYZ_FORMAT          udiff | udiff-h | udiff-l | search-replace, default 'search-replace'
    DIFFXYZ_SYSTEM          none | format, default 'format'
    DIFFXYZ_LIMIT           int, default 20
    DIFFXYZ_LANG            single language for the eval split, default 'python'
    DIFFXYZ_SEED            int, default 0
    DIFFXYZ_EVAL_MAX_TOKENS int, default 512
    DIFFXYZ_OUT             output JSON path, default '/workspace/project/result.json'

Required keys (loaded from /workspace/project/.env at startup so the
env_forward denylist on HF_TOKEN doesn't matter):
    HF_TOKEN              for gated / rate-limited HF Hub fetches
    WANDB_API_KEY         optional, only used if WANDB_PROJECT is set

Stdout contract: per-sample line + final ``RESULT EM=<f> IoU=<f>`` line +
self-archived ``RESULT_JSON_B64_*`` envelope so collect can recover the
full result.json from the log even after the pod is destroyed.
"""
from __future__ import annotations

import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Make `evaluation.diff_xyz` importable when this script is run from the tap root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_env_file_if_present() -> None:
    """Load /workspace/project/.env (or ./.env) into os.environ.

    Crucible's env_forward denylist blocks HF_TOKEN from reaching the
    launcher process directly, but `.env.runpod.local` is rsynced as
    `/workspace/project/.env` on the pod. Read it once so newer
    huggingface_hub picks up the token and stops rate-limiting.
    """
    candidates = [
        Path("/workspace/project/.env"),
        Path.cwd() / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        except OSError:
            continue
        return


def _strip_ws_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def _first_diverging_line(predicted: str, reference: str) -> tuple[int, str, str] | None:
    """Return (line_index, pred_line, ref_line) for the first non-matching
    line after stripping whitespace-only lines (matches `stripped_em` rule).

    Returns None if the two are identical post-strip (i.e. EM should be 1.0).
    """
    p = _strip_ws_lines(predicted)
    r = _strip_ws_lines(reference)
    for i in range(min(len(p), len(r))):
        if p[i] != r[i]:
            return (i, p[i], r[i])
    if len(p) != len(r):
        idx = min(len(p), len(r))
        pred_line = p[idx] if idx < len(p) else "<EOF>"
        ref_line = r[idx] if idx < len(r) else "<EOF>"
        return (idx, pred_line, ref_line)
    return None


def main() -> int:
    _load_env_file_if_present()

    base_model = os.environ.get("DIFFXYZ_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    task = os.environ.get("DIFFXYZ_TASK", "diff_gen")
    fmt = os.environ.get("DIFFXYZ_FORMAT", "search-replace")
    sys_mode = os.environ.get("DIFFXYZ_SYSTEM", "format")
    limit = int(os.environ.get("DIFFXYZ_LIMIT", "20"))
    lang = os.environ.get("DIFFXYZ_LANG", "python")
    seed = int(os.environ.get("DIFFXYZ_SEED", "0"))
    max_tokens = int(os.environ.get("DIFFXYZ_EVAL_MAX_TOKENS", "512"))
    out_path = Path(os.environ.get("DIFFXYZ_OUT", "/workspace/project/result.json"))

    print(f"[eval_only] base={base_model} task={task} fmt={fmt} sys={sys_mode} "
          f"limit={limit} lang={lang} max_tokens={max_tokens}", flush=True)

    from evaluation.diff_xyz.dataset import load_samples
    from evaluation.diff_xyz.harness import score_sample
    from evaluation.diff_xyz.models import HFBackend

    print(f"[eval_only] loading {base_model} in 4-bit...", flush=True)
    backend = HFBackend(model_id=base_model, load_in_4bit=True)
    # Force the lazy-load now so any auth/network failure surfaces before
    # we start iterating samples.
    backend._load()

    samples = load_samples(limit=limit, langs=[lang], seed=seed)
    print(f"[eval_only] eval {len(samples)} {lang} samples...", flush=True)

    rows = []
    t_total = time.time()
    for i, sample in enumerate(samples):
        t1 = time.time()
        r = score_sample(backend, sample, idx=i, task=task, fmt=fmt,
                         sys_mode=sys_mode, max_tokens=max_tokens, temperature=0.0)
        rows.append(r)
        if task == "diff_gen":
            print(f"[eval_only] eval[{i+1}/{len(samples)}] em={r.em:.2f} "
                  f"iou={r.iou:.2f} parse={r.parsing_rate:.0f} apply={r.applying_rate:.0f} "
                  f"f1+={r.f1_plus:.2f} f1-={r.f1_minus:.2f} t={time.time()-t1:.1f}s",
                  flush=True)
        else:
            print(f"[eval_only] eval[{i+1}/{len(samples)}] em={r.em:.2f} "
                  f"iou={r.iou:.2f} t={time.time()-t1:.1f}s", flush=True)
        # Live divergence diagnostic: surface the first non-matching line on
        # near-misses (parsed + applied but not byte-exact). Helps spot
        # whitespace/format-spec drift without decoding the base64 archive.
        if r.em == 0.0 and r.applying_rate == 1.0 and r.predicted and r.reference:
            diff = _first_diverging_line(r.predicted, r.reference)
            if diff is not None:
                idx, pred_line, ref_line = diff
                pred_line = pred_line[:80]
                ref_line = ref_line[:80]
                print(f"[diag][{i}] first_diff line={idx} "
                      f"pred={pred_line!r} ref={ref_line!r}", flush=True)

    em = statistics.fmean([r.em for r in rows]) if rows else 0.0
    iou = statistics.fmean([r.iou for r in rows]) if rows else 0.0

    metrics: dict = {"EM": em, "IoU": iou, "n": len(rows)}
    if task == "diff_gen":
        metrics["parsing_rate"] = statistics.fmean([r.parsing_rate for r in rows]) if rows else 0.0
        metrics["applying_rate"] = statistics.fmean([r.applying_rate for r in rows]) if rows else 0.0
        metrics["f1_plus"] = statistics.fmean([r.f1_plus for r in rows]) if rows else 0.0
        metrics["f1_minus"] = statistics.fmean([r.f1_minus for r in rows]) if rows else 0.0

    payload = {
        "mode": "eval_only",
        "base_model": base_model,
        "task": task,
        "fmt": fmt,
        "sys_mode": sys_mode,
        "lang": lang,
        "limit": limit,
        "max_tokens": max_tokens,
        "metrics": metrics,
        "elapsed_s": round(time.time() - t_total, 2),
        "per_sample": [
            {"idx": r.idx, "em": r.em, "iou": r.iou, "lang": r.lang,
             "parsing_rate": r.parsing_rate, "applying_rate": r.applying_rate,
             "f1_plus": r.f1_plus, "f1_minus": r.f1_minus,
             "elapsed_s": r.elapsed_s, "error": r.error,
             "predicted": r.predicted, "reference": r.reference}
            for r in rows
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eval_only] wrote {out_path}", flush=True)

    blob = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    print("RESULT_JSON_B64_BEGIN", flush=True)
    for chunk in (blob[i:i + 76] for i in range(0, len(blob), 76)):
        print(chunk, flush=True)
    print("RESULT_JSON_B64_END", flush=True)
    print(f"RESULT EM={em:.4f} IoU={iou:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
