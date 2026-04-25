# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Diff-XYZ Colab Smoke
#
# Runs the JetBrains Diff-XYZ benchmark (arXiv 2510.12487) against an LLM API
# (OpenAI, Anthropic, or Gemini) and prints EM / IoU / F1 metrics.
#
# **Setup:**
# 1. Runtime: free CPU tier is fine — this uses APIs, not local models.
# 2. Open the key-vault icon on the left sidebar → add `OPENAI_API_KEY`,
#    `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY` (whichever you use).
#    Set "Notebook access" ON for each.
# 3. Edit the **Config** cell below to pick model / task / format / limit.
# 4. Run all cells.
#
# **Reference (paper Table 1, w/format system prompt):**
# - GPT-4.1-mini Apply udiff → EM 0.90
# - Claude 4 Sonnet Apply udiff → EM 0.96
# - GPT-4.1 Diff-Gen search-replace → EM 0.95
#
# If your 20-sample result is off by more than ±5pp from these, the prompts
# may need to match paper Appendix A more exactly.

# %% [markdown]
# ## 1. Install dependencies

# %%
import os, subprocess
REPO_URL = "https://github.com/eren23/crucible-community-tap"
REPO_DIR = "/content/crucible-community-tap"
if not os.path.isdir(REPO_DIR) or not os.listdir(REPO_DIR):
    subprocess.run(f"git clone --depth 1 {REPO_URL} {REPO_DIR}", shell=True, check=True)
os.chdir(REPO_DIR)
print("cwd:", os.getcwd())
print("commit:", subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True, text=True).stdout.strip())

# Tap is plain Python (no pyproject); pip-install only the deps we need.
subprocess.run("pip install -q datasets openai anthropic google-generativeai",
               shell=True, check=False)

# %% [markdown]
# ## 3. Load API keys from Colab secrets

# %%
import os
try:
    from google.colab import userdata
except ImportError:
    userdata = None

for _key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN"):
    if os.environ.get(_key):
        continue
    if userdata is None:
        continue
    try:
        _v = userdata.get(_key)
    except Exception:
        _v = None
    if _v:
        os.environ[_key] = _v

_loaded = sorted(k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY") if os.environ.get(k))
print("API keys loaded:", _loaded or "(none — add one via the left sidebar key icon)")

# %% [markdown]
# ## 4. Config — edit me

# %%
# Pick one of: "openai:gpt-4.1-mini", "openai:gpt-4.1", "anthropic:claude-sonnet-4-5",
#              "anthropic:claude-haiku-4-5-20251001", "google:gemini-2.5-flash"
MODEL = "openai:gpt-4.1-mini"

# Pick one of: "apply" (old+diff→new), "anti_apply" (new+diff→old), "diff_gen" (old+new→diff)
TASK = "apply"

# Pick one of: "udiff", "udiff-h", "udiff-l", "search-replace"
FORMAT = "udiff"

# "format" (system prompt describes the diff format) or "none" (generic system prompt)
SYSTEM_PROMPT = "format"

# Number of samples. 20 is fast (~30s + API latency). Full benchmark is 1000.
LIMIT = 20

# Optional: restrict to specific languages. Empty list = all 5 langs.
LANGS = []  # e.g. ["python"] or ["python", "rust"]

OUT_PATH = "/content/diff_xyz_result.json"
SEED = 0

print(f"config: model={MODEL}  task={TASK}  format={FORMAT}  system={SYSTEM_PROMPT}  limit={LIMIT}")

# %% [markdown]
# ## 5. Run the benchmark

# %%
import os, subprocess, sys, time

cmd = [
    sys.executable, "-m", "evaluation.diff_xyz.harness",
    "--model", MODEL,
    "--task", TASK,
    "--format", FORMAT,
    "--system-prompt", SYSTEM_PROMPT,
    "--limit", str(LIMIT),
    "--seed", str(SEED),
    "--out", OUT_PATH,
]
if LANGS:
    cmd += ["--langs", *LANGS]

print("$", " ".join(cmd))
t0 = time.time()
proc = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    env=os.environ,
)
for _line in proc.stdout:
    sys.stdout.write(_line); sys.stdout.flush()
proc.wait()
print(f"\n--- exit={proc.returncode}  elapsed={time.time()-t0:.1f}s ---")
if proc.returncode != 0:
    raise RuntimeError(f"harness exited with code {proc.returncode}")

# %% [markdown]
# ## 6. Results summary

# %%
import json
with open(OUT_PATH) as f:
    result = json.load(f)

print(f"Model:   {result['model']}")
print(f"Task:    {result['task']} / {result['format']} / system={result['system_prompt']}")
print(f"Samples: {result['n_samples']}")
print()
print("Overall metrics:")
for k, v in result["metrics"].items():
    if isinstance(v, float):
        print(f"  {k:18s} {v:.4f}")
    else:
        print(f"  {k:18s} {v}")

if result.get("per_lang"):
    print("\nPer-language:")
    for lang, m in sorted(result["per_lang"].items()):
        em = m.get("EM", 0.0)
        iou = m.get("IoU", 0.0)
        print(f"  {lang:12s}  EM={em:.3f}  IoU={iou:.3f}  n={sum(1 for r in result['per_sample'] if r['lang']==lang)}")

# %% [markdown]
# ## 7. Inspect individual samples (optional)

# %%
# Print the first few per-sample rows to debug low scores.
import json
for row in result["per_sample"][:5]:
    err = f"  error={row['error'][:80]}" if row.get("error") else ""
    print(
        f"sample[{row['idx']:3d}] lang={row['lang']:10s} "
        f"em={row['em']:.2f} iou={row['iou']:.2f} "
        f"chars={row['response_chars']:5d} t={row['elapsed_s']:.2f}s{err}"
    )

# %% [markdown]
# ## 8. (Optional) Save / upload result JSON
#
# Uncomment the Google Drive cell to persist the result for offline analysis.

# %%
# from google.colab import drive
# drive.mount("/content/drive")
# import shutil
# shutil.copy(OUT_PATH, "/content/drive/MyDrive/diff_xyz_result.json")
# print("saved to Drive")

# %% [markdown]
# ## Next steps
#
# - **Sweep formats**: change `FORMAT` to `"search-replace"` — paper shows this wins for
#   larger models on Diff-Gen (GPT-4.1 search-replace Diff-Gen EM 0.95 vs udiff 0.76).
# - **Full benchmark**: set `LIMIT = 1000`. Takes ~20 min wall time per task/format combo.
# - **Different model**: try `"anthropic:claude-sonnet-4-5"` for a frontier baseline.
# - **Per-sample debug**: inspect `result["per_sample"]` to see which languages / hunk
#   counts / change kinds the model drops on.
