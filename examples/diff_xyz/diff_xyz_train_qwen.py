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
# # Fine-tune Qwen3-Coder on Diff-XYZ (LoRA + 4-bit)
#
# End-to-end pipeline:
# 1. Load **Qwen3-Coder-30B-A3B-Instruct** (MoE, 3B active, ~15GB in 4-bit → fits H100 80GB easily)
#    or **Qwen3-Coder-Next (80B dense)** for the tighter H100 option.
# 2. Build SFT data from CommitPackFT-python, **deduped against Diff-XYZ test** by `(repo, commit, path)`.
#    Each commit produces 3 training examples (Apply, Anti-Apply, Diff-Gen), in **search-replace** format.
# 3. LoRA SFT with TRL on ~2000 examples, ~500 steps.
# 4. Evaluate on a Diff-XYZ subset (Python, 50 samples, Apply task) and print lift vs base.
#
# ### Runtime requirements
#
# | Model | VRAM (4-bit) | Colab runtime |
# |-------|--------------|----------------|
# | Qwen3-Coder-30B-A3B-Instruct (default) | ~15GB | H100 80GB, A100 40GB |
# | Qwen3-Coder-Next (80B dense) | ~40GB | H100 80GB only (tight) |
# | Qwen2.5-Coder-7B-Instruct (fallback) | ~5GB | T4 (free) + small LIMIT_TRAIN |
#
# Data: CommitPackFT Python subset (~50k rows) → filter → ~2k training examples.
# Wall time: 30–60 min on H100 for defaults.

# %% [markdown]
# ## 1. Install deps

# %%
import subprocess
subprocess.run(
    "pip install -q 'transformers>=4.50' 'accelerate>=0.30' 'bitsandbytes>=0.43' "
    "'peft>=0.11' 'trl>=0.9' datasets huggingface_hub wandb",
    shell=True, check=False,
)
# flash-attn is optional; skip if install fails (falls back to SDPA).
_rc = subprocess.run(
    "pip install -q flash-attn --no-build-isolation", shell=True, check=False
).returncode
print("flash-attn:", "installed" if _rc == 0 else "not installed (using SDPA)")

# %% [markdown]
# ## 2. Clone the tap (harness + formats + metrics) and install in editable mode

# %%
import os, subprocess
REPO_URL = "https://github.com/eren23/crucible-community-tap"
REPO_DIR = "/content/crucible-community-tap"
if not os.path.isdir(REPO_DIR) or not os.listdir(REPO_DIR):
    subprocess.run(f"git clone --depth 1 {REPO_URL} {REPO_DIR}", shell=True, check=True)
os.chdir(REPO_DIR)
print("cwd:", os.getcwd())
print("commit:", subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True, text=True).stdout.strip())

# `evaluation.diff_xyz` is importable from the tap root once we add it to sys.path.
import sys
sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 3. Secrets (HF_TOKEN for gated models, WANDB_API_KEY for logging)

# %%
import os
try:
    from google.colab import userdata
except ImportError:
    userdata = None
for _key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY"):
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
print("secrets:", sorted(k for k in ("HF_TOKEN", "WANDB_API_KEY") if os.environ.get(k)) or "(none)")

# %% [markdown]
# ## 4. Config — edit me

# %%
# --- Model choice ---
# Recommended for H100 80GB:
MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"   # MoE, 3B active, ~15GB @ 4-bit
# Tight H100 80GB option (80B dense, LoRA-only, ~40GB @ 4-bit):
# MODEL_ID = "Qwen/Qwen3-Coder-Next"
# Free T4 fallback (~5GB @ 4-bit, drops SOTA but proves the pipeline):
# MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

# --- Data ---
TRAIN_DATASET = "bigcode/commitpackft"        # Python subset only; see filter below
TRAIN_LANG = "python"
LIMIT_TRAIN = 2000                            # number of COMMITS (×3 tasks = 6k training examples)
EVAL_LIMIT = 50                               # Diff-XYZ samples to score post-training
EVAL_LANG = "python"

# --- LoRA ---
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]  # add mlp layers for more capacity

# --- Training ---
MAX_STEPS = 500
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
LR = 2e-5
MAX_SEQ_LEN = 4096                            # commits rarely need more
WARMUP_RATIO = 0.03
SEED = 42

# --- Logging ---
WANDB_PROJECT = "crucible-diff-xyz-qwen"
WANDB_RUN_NAME = "qwen3-coder-30b-a3b-lora-v1"

# --- Paths ---
OUTPUT_DIR = "/content/ckpts/qwen-diffxyz-lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"model:  {MODEL_ID}\ntrain:  {LIMIT_TRAIN} commits\neval:   {EVAL_LIMIT} Diff-XYZ samples")

# %% [markdown]
# ## 5. Load base model (4-bit) + LoRA adapter

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"loading {MODEL_ID} in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" if _rc == 0 else "sdpa",
)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# %% [markdown]
# ## 6. Load + dedupe training data (CommitPackFT − Diff-XYZ test)

# %%
from datasets import load_dataset

print("loading Diff-XYZ test (for dedup)...")
dxyz = load_dataset("JetBrains-Research/diff-xyz", split="test")
dxyz_keys: set[tuple] = {
    (row["repo"], row["commit"], row["path"]) for row in dxyz
}
print(f"Diff-XYZ test: {len(dxyz_keys)} unique (repo, commit, path) tuples")

print(f"loading {TRAIN_DATASET} / {TRAIN_LANG}...")
raw = load_dataset(TRAIN_DATASET, TRAIN_LANG, split="train", streaming=True)

def _not_in_test(row):
    key = (row.get("repos", row.get("repo", "")),
           row.get("commit", ""),
           row.get("old_file", row.get("path", "")))
    return key not in dxyz_keys

filtered = []
skipped = 0
for row in raw:
    if len(filtered) >= LIMIT_TRAIN:
        break
    # CommitPackFT has old_contents, new_contents, subject, message, old_file, repos, commit.
    if not row.get("old_contents") or not row.get("new_contents"):
        continue
    if not _not_in_test(row):
        skipped += 1
        continue
    filtered.append({
        "repo": row.get("repos", ""),
        "commit": row.get("commit", ""),
        "path": row.get("old_file", ""),
        "old_code": row["old_contents"],
        "new_code": row["new_contents"],
        "message": row.get("message", ""),
    })

print(f"training commits: {len(filtered)}  (skipped {skipped} in Diff-XYZ test)")

# %% [markdown]
# ## 7. Build SFT examples (multi-task × search-replace)

# %%
from evaluation.diff_xyz.prompts import GENERIC_SYSTEM, system_prompt, user_prompt
from evaluation.diff_xyz.dataset import DiffXYZSample
import difflib

def _build_search_replace(old: str, new: str) -> str:
    """Produce a search-replace diff from two snippets via line-level matching."""
    old_lines = old.splitlines(keepends=False)
    new_lines = new.splitlines(keepends=False)
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    blocks = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        search = "\n".join(old_lines[i1:i2])
        replace = "\n".join(new_lines[j1:j2])
        blocks.append(f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE")
    return "\n".join(blocks) if blocks else ""

examples: list[dict] = []
for row in filtered:
    sr = _build_search_replace(row["old_code"], row["new_code"])
    if not sr:
        continue
    sample = DiffXYZSample(
        repo=row["repo"], commit=row["commit"], path=row["path"], lang=TRAIN_LANG,
        old_code=row["old_code"], new_code=row["new_code"],
        udiff="", udiff_h="", udiff_l="", search_replace=sr,
        n_added=0, n_removed=0, n_hunks=0, change_kind="",
    )
    for task, target in [
        ("apply",       row["new_code"]),
        ("anti_apply",  row["old_code"]),
        ("diff_gen",    sr),
    ]:
        sys_p = system_prompt("search-replace", "format")
        usr_p = user_prompt(task, "search-replace", sample)
        examples.append({
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
                {"role": "assistant", "content": target},
            ],
        })

print(f"SFT examples: {len(examples)}  (apply + anti_apply + diff_gen per commit)")

# %% [markdown]
# ## 8. Tokenize + train

# %%
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

ds = Dataset.from_list(examples)

sft_cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    max_seq_length=MAX_SEQ_LEN,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    seed=SEED,
    report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
    run_name=WANDB_RUN_NAME,
    gradient_checkpointing=True,
    packing=False,
)

if os.environ.get("WANDB_API_KEY"):
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=ds,
    processing_class=tokenizer,
)
print("starting training...")
trainer.train()
print("saving adapter...")
trainer.save_model(OUTPUT_DIR)
print(f"saved to {OUTPUT_DIR}")

# %% [markdown]
# ## 9. Post-training eval (Diff-XYZ Apply, Python subset)

# %%
# In-process eval using the fine-tuned adapter directly on the already-loaded model.
# We reuse the harness's scoring logic without shelling out.

from evaluation.diff_xyz.dataset import load_samples
from evaluation.diff_xyz.models import HFBackend
from evaluation.diff_xyz.harness import score_sample
import statistics

# Free training allocator noise before eval.
import gc, torch
gc.collect(); torch.cuda.empty_cache()

eval_samples = load_samples(limit=EVAL_LIMIT, langs=[EVAL_LANG], seed=SEED)
print(f"evaluating {len(eval_samples)} samples from Diff-XYZ ({EVAL_LANG})")

# Wrap the trained adapter in an HFBackend that shares the loaded model.
ft_backend = HFBackend(model_id=MODEL_ID)
ft_backend._model = model            # skip reload
ft_backend._tokenizer = tokenizer

EVAL_TASK = "apply"
EVAL_FORMAT = "search-replace"
EVAL_SYS = "format"

rows = []
for i, sample in enumerate(eval_samples):
    r = score_sample(ft_backend, sample, idx=i, task=EVAL_TASK, fmt=EVAL_FORMAT,
                     sys_mode=EVAL_SYS, max_tokens=2048, temperature=0.0)
    rows.append(r)
    if (i + 1) % 10 == 0:
        em = statistics.fmean([r.em for r in rows])
        print(f"[{i+1}/{len(eval_samples)}] running EM={em:.3f}")

em = statistics.fmean([r.em for r in rows]) if rows else 0.0
iou = statistics.fmean([r.iou for r in rows]) if rows else 0.0
print(f"\n=== Fine-tuned {MODEL_ID} ({EVAL_TASK}/{EVAL_FORMAT}) ===")
print(f"EM:  {em:.4f}")
print(f"IoU: {iou:.4f}")
print(f"n:   {len(rows)}")

# %% [markdown]
# ## 10. Compare to base model (optional, ~10 min on H100)

# %%
# Disable the LoRA adapter to get base-model numbers for the same samples.
# (Re-enable is as easy as `model.enable_adapter_layers()`.)

RUN_BASE_COMPARE = True  # set to False to skip

if RUN_BASE_COMPARE:
    print("disabling adapter — evaluating base model on same samples...")
    model.disable_adapter_layers()
    base_rows = []
    for i, sample in enumerate(eval_samples):
        r = score_sample(ft_backend, sample, idx=i, task=EVAL_TASK, fmt=EVAL_FORMAT,
                         sys_mode=EVAL_SYS, max_tokens=2048, temperature=0.0)
        base_rows.append(r)
        if (i + 1) % 10 == 0:
            em = statistics.fmean([r.em for r in base_rows])
            print(f"[base {i+1}/{len(eval_samples)}] EM={em:.3f}")
    model.enable_adapter_layers()
    base_em = statistics.fmean([r.em for r in base_rows]) if base_rows else 0.0
    base_iou = statistics.fmean([r.iou for r in base_rows]) if base_rows else 0.0
    print("\n=== Comparison ===")
    print(f"Base  EM={base_em:.4f}  IoU={base_iou:.4f}")
    print(f"FT    EM={em:.4f}  IoU={iou:.4f}")
    print(f"Δ EM  {em - base_em:+.4f}  Δ IoU  {iou - base_iou:+.4f}")

# %% [markdown]
# ## 11. (Optional) Push adapter to Hugging Face Hub

# %%
PUSH_TO_HUB = False
HUB_REPO_ID = "your-username/qwen3-coder-diffxyz-lora"

if PUSH_TO_HUB and os.environ.get("HF_TOKEN"):
    from huggingface_hub import upload_folder
    upload_folder(
        repo_id=HUB_REPO_ID, folder_path=OUTPUT_DIR,
        repo_type="model", token=os.environ["HF_TOKEN"],
        commit_message=f"LoRA adapter trained on Diff-XYZ (search-replace, {LIMIT_TRAIN} commits, {MAX_STEPS} steps)",
    )
    print(f"pushed to https://huggingface.co/{HUB_REPO_ID}")

# %% [markdown]
# ## What's next
#
# - **Scale up data**: `LIMIT_TRAIN = 10000` + `MAX_STEPS = 2000` for a stronger model (~4h on H100).
# - **Full Diff-XYZ eval**: `EVAL_LIMIT = 200` and `EVAL_LANG = None` to hit all 1000 samples across 5 langs.
# - **Sweep format**: retrain with `udiff` instead of `search-replace` — compare format-choice under SFT.
# - **Target Diff-Gen directly**: `EVAL_TASK = "diff_gen"` — that's where frontier's real headroom lives (Claude 4 Sonnet 0.94 → aim for ≥0.96 with this open model).
