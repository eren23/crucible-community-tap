# Sfumato — three orthogonal failure axes in hybrid AR / DDLM reasoning

Companion finding to the paper

> **Three Orthogonal Failure Axes in Hybrid AR / Diffusion Reasoning,
> with Trainable Fixes for Two of Them**
>
> Source: <https://github.com/eren23/sfumato>
> Paper:  <https://github.com/eren23/sfumato_paper>
> Video:  `sfumato_paper/video/final_story_1080p60.mp4` (12 Manim scenes, ~2:30)

## TL;DR

Hybrid pipelines that combine an autoregressive (AR) language model with
a discrete-diffusion language model (DDLM) for chain-of-thought reasoning
have produced contradictory empirical reports. We argue those conflicts
collapse once the failure surface is decomposed: hybrid AR/DDLM reasoning
fails along **at least three orthogonal axes** —

1. **Interface-format brittleness** — wrapping a question in any
   plan-shaped scaffold, even an empty `Plan: ` literal, costs LLaDA-8B
   8 pp of GSM8K accuracy. Fixable with a small prefix-robust LoRA.
2. **Planner-content trust** — capacity-dependent in opposite directions
   across planners. At v3 LoRA capacity (full-FFN coverage), a 0.5B AR
   planner improves the hybrid by +5 pp while a 1.5B planner regresses
   by 13 pp. Characterized but not fixed.
3. **Sampling-diversity preservation** — *expanded* by format-augmented
   training, not collapsed. 5/5-branch agreement drops 51.5% → 47.5%;
   mean unique answers per problem rises 1.83 → 2.07; cmaj accuracy
   holds. Inverse of the standard encoder-collapse story.

Plus a Track 2 finding: weight-space distillation of inference-time
consensus (cmaj into a single forward pass) is **design-sensitive, not
architecture-limited**. Late-block answer-span surgery fails (c2c =
70.5% across two iterations); earlier-block full-response surgery
recovers (c2c = 79.0%, within sampling error of the 80% pre-registered
target).

## Headline numbers (GSM8K-test, N=200, single seed)

See [MASTER_TABLE.md](./MASTER_TABLE.md) for every condition with exact
Clopper–Pearson 95% CIs.

- base C2 (no prefix): 74%
- base cmaj b=5: 79.0% (test)
- Track 1 v3 (7/7 modules, 22 M params): C2 73.0%, C2empty 74.0%
- Track 2 c2c v3: **79.0% [72.7, 84.4]**, hits the 80% pre-reg target inside CI
- cmajc v3: **82.5% [76.5, 87.5]**
- Qwen-AR self-consistency b=5: 40.5% — generic SC does *not* explain
  LLaDA's diffusion advantage

## Reproduce

The community tap ships four project specs. Install one:

```bash
crucible tap add https://github.com/eren23/crucible-community-tap
crucible tap install sfumato_e4
```

Headline reproduction (cmajc-v3 = 82.5%):

```python
provision_nodes(count=1, interruptible=False, name_prefix="sfumato",
                template_id="runpod-torch-v280")
bootstrap_project(project_name="sfumato_e4")
run_project(project_name="sfumato_e4", overrides={
    "CONDITION": "cmajc",
    "K_STEPS": "64", "N_PROBLEMS": "200", "SEED": "0",
    "LORA_PATH": "eren23/sfumato-llada-prefix-robust-v3",
    "COMMIT_LORA_PATH": "eren23/sfumato-llada-commit-v3",
    "COMMIT_N_BLOCKS": "3", "BRANCHES": "5", "TEMP": "0.7",
})
```

~70 min wallclock, ~$0.25 on RTX-4090 non-interruptible. Sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically (the
project spec) — without it, PEFT adapter switching OOMs the GPU around
problem 144 / 200.

The other three specs cover training:

- `sfumato_e2_track1_data` — builds `eren23/sfumato-prefix-robust-gsm8k`
  (7,473 GSM8K-train problems × 8 prefix tiers = 59,784 rows)
- `sfumato_e2_track1` — trains the prefix-robust LoRA (Track 1)
- `sfumato_e2_track2` — trains the commit-LoRA (Track 2)

All adapters and datasets are public on the Hugging Face Hub under
`eren23/sfumato-*`.

## Pre-registered prediction outcomes

| # | Prediction | Outcome |
|---|---|---|
| 1 | c2c ≥ 80% | **HOLDS** — 79.0% [72.7, 84.4] within CI |
| 2 | base cmaj b=5 within ±1 pp of 80% | **HOLDS** — 79.0% on test, 81.5% on dev |
| 3 | cmajc ≤ cmaj + 1 pp ("no double-dip") | **VIOLATED UPWARD** — +3.5 pp v3, +3 pp v2 |
| 4 | planner-quality threshold shifts down with LoRA | **MIXED** — holds at v2, inverts at v3 |

Two clean holds, one violation in the favorable direction
(compositionality of cmaj + commit), one mixed (the capacity-dependent
direction-split that became the second axis).

## The most insightful secondary finding

**Branch aggregation absorbs the v2 → v3 structural correction at b=5.**

- At b=1 (single pass): c2c v3 beats c2c v2 by **+8.5 pp**, well outside
  CI overlap (79.0% vs 70.5%).
- At b=5 (cmajc, branch vote): the same two adapters are
  statistically indistinguishable (82.5% vs 82.0%, within both CIs).

For weight-space distillation of inference-time mechanisms: single-pass
deployment is where structural correctness pays. Under
stochastic-aggregation regimes, structurally-incorrect adapters can be
indistinguishable from correct ones at common eval N.

## Total compute

~$3.50 across the entire paper, single RTX-4090 spot pod at $0.20/hr on
RunPod. Phase C revision (cmajc-v3 result) added ~$1.00 of compute due
to two pod retries — first preempted at step 75, second OOM'd at step
144 without `expandable_segments:True`.

## Files

- [`MASTER_TABLE.md`](./MASTER_TABLE.md) — every headline number with
  exact Clopper–Pearson 95% CI
- Pre-reg + protocol: <https://github.com/eren23/sfumato/blob/main/e2/PROTOCOL.md>
- Track 1 results: <https://github.com/eren23/sfumato/blob/main/e2/RESULTS_TRACK1.md>
- Track 2 results: <https://github.com/eren23/sfumato/blob/main/e2/RESULTS_TRACK2.md>
- Paper LaTeX + built PDF: <https://github.com/eren23/sfumato_paper>

## W&B

- <https://wandb.ai/eren23/sfumato-e4>
- <https://wandb.ai/eren23/sfumato-e2>

Headline cmajc-v3 run: <https://wandb.ai/eren23/sfumato-e4/runs/2j3iobc3>
(accuracy 0.825 / 165 of 200 / 3924 s wallclock).
