# CodeWM paper baseline rerun — master finding (2026-04-10 → ongoing)

A running record of every eval rerun and ablation performed for the
CodeWM paper, post the silent `WM_POOL_MODE=cls` bug fix in tap commit
`de81267` (2026-04-10). Each section corresponds to a distinct eval
script or ablation; sections are append-only so the historical
progression is preserved even as the paper narrative evolves.

## Sections

1. `codebert_compare.py` — in-distribution CommitPackFT retrieval (n=200/500)
2. `cross_repo_modern_compare.py` (9-repo, SUPERSEDED by §8)
3. `codesearchnet_eval.py` — OOD function-level retrieval (n=300/1000)
4. `modern_baselines_compare.py` (t5 venv) — CodeSearchNet head-to-head
5. `honest_retrieval.py` — in-distribution honest retrieval (n=500/2000)
6. `modern_baselines_compare.py` (t4 venv) — jina + codet5p added
7. `commit_chronicle_eval.py` — SUPERSEDED by §9 (exposed as saturation artefact)
8. **Track A + D (2026-04-11)** — cross-repo 20-repo rerun, class-prior chance, hard-neg K=9 (reverses CodeBERT win into CodeWM win)
9. **Track B (2026-04-11)** — commit-chronicle subset_cmg Python rerun (downgrades subset_llm "second win" to saturation artefact)
10. **Exp 2 (2026-04-11)** — delta norm / magnitude reporting (kills the "tiny vector" reviewer concern)
11. **Exp 1 (2026-04-11, in progress)** — frozen-target ablation (`WM_EMA_DECAY=1.0`, seeds 42+43)

## Links

- [Full table with all numbers and narrative](./MASTER_TABLE.md)
- [Paper source (private): `~/Documents/AI/codewm-paper/paper/main.tex`]
- [Spider Chat session note: `CodeWM — Eval Bug Sweep & Multi-seed Replication`]
- [Config hierarchy reference: `parameter-golf_dev/docs/crucible-config-hierarchy.md`]
- [delta_norm_report.py script: `evaluation/code_wm/delta_norm_report.py`]

## Provenance

Originally lived at `/tmp/codewm_eval_results/paper_rerun/MASTER_TABLE.md`
on the author's Mac. Copied into the tap at
`findings/code_wm_paper_rerun/MASTER_TABLE.md` on 2026-04-11 so that
reboots don't wipe it and so it can be pushed to the tap git remote
as part of the public reproducibility trail. The /tmp copy remains
the live working copy during a session (edited by the eval scripts
and by the author), and is copied back into the tap's findings/
directory at the end of each session. Keep them in sync — last
paragraph of this README should be updated to mention the copy date
whenever the /tmp file is newer.

**Last sync**: 2026-04-11 (post frozen-target ablation, Exp 1 complete). Section 11 filled in with real numbers. Scenario A+ confirmed — val_dcos matches baseline within ±0.002, retrieval matches or exceeds baseline, but the frozen-target latent geometry has 40-80% larger delta magnitudes than the near-frozen EMA baseline (unexpected secondary finding).
