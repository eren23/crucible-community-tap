# Multi-step Rollout Results (2026-04-05)

Raw numbers from `evaluation/rollout_eval.py trajectory` run against two checkpoints
on parameter-golf_dev git history. 51 trajectories total (20 with 5 states, 31 with 4 states).

## Setup

- **Test repo**: parameter-golf_dev (1 Python repo, ~5000 commits, various file sizes)
- **Trajectory extraction**: `git log --no-merges`, group by file, take newest N commits per file
- **Trajectories filtered** to files with ≥ N sequential edits (N=4 or 3)
- **States**: each trajectory has k+1 ordered file states, k transitions
- **Actions**: 7-dim structural diffs via `collectors/ast_diff.compute_rich_action(before, after)[:7]`
- **Metrics**: all cosine similarities in 128d (G10) or 192d (ExpA) latent space
- **Rolled-out**: predictor composed on its own outputs. z_t = predict(z_{t-1}, action_t)
- **Teacher-forced**: predictor fed ground-truth encoding at each step
- **Copy-last**: trivial baseline — predict z_{t+1} = z_t

## Results

### 4-step trajectories (20 trajs, parameter-golf_dev)

**G10 (128d SIGReg+dir1.0)**
| step | teach_cos | roll_cos | copy_cos | vs_copy  | delta_cos |
|------|:---------:|:--------:|:--------:|:--------:|:---------:|
|  1   |  0.9864   |  0.9864  |  0.9747  | +0.0117  |  0.1372   |
|  2   |  0.9646   |  0.9639  |  0.9857  | -0.0218  |  0.0090   |
|  3   |  0.8695   |  0.8682  |  0.9055  | -0.0373  | -0.0127   |
|  4   |  0.8720   |  0.8706  |  0.9999  | -0.1293  | -0.0042   |

**ExpA (192d G8 recipe)**
| step | teach_cos | roll_cos | copy_cos | vs_copy  | delta_cos |
|------|:---------:|:--------:|:--------:|:--------:|:---------:|
|  1   |  0.9954   |  0.9954  |  0.9345  | **+0.0609** |  0.1595   |
|  2   |  0.9993   |  0.9991  |  0.9982  | +0.0009  |  0.0324   |
|  3   |  0.9995   |  0.9994  |  0.9999  | -0.0005  | -0.0995   |
|  4   |  0.9988   |  0.9986  |  0.9997  | -0.0011  | -0.0527   |

### 3-step trajectories (31 trajs, parameter-golf_dev)

**G10 (128d)**
| step | teach_cos | roll_cos | copy_cos | vs_copy  | delta_cos |
|------|:---------:|:--------:|:--------:|:--------:|:---------:|
|  1   |  0.9742   |  0.9742  |  0.9869  | -0.0127  |  0.0859   |
|  2   |  0.9156   |  0.9148  |  0.9409  | -0.0261  | -0.0111   |
|  3   |  0.9168   |  0.9158  |  1.0000  | -0.0842  | -0.0021   |

**ExpA (192d)**
| step | teach_cos | roll_cos | copy_cos | vs_copy  | delta_cos |
|------|:---------:|:--------:|:--------:|:--------:|:---------:|
|  1   |  0.9994   |  0.9994  |  0.9552  | **+0.0442** |  0.1187   |
|  2   |  0.9949   |  0.9948  |  0.9956  | -0.0008  | -0.0056   |
|  3   |  0.9925   |  0.9922  |  0.9916  | +0.0006  | -0.0160   |

## Summary

- **ExpA > G10** on step-1 lift (5x stronger on 20-traj, unstable on 31-traj for G10)
- **Both fail step 2+**: rolled-out rarely beats copy-last, and when it does the margin is ≤0.001
- **delta direction alignment** collapses to noise by step 2 for both models
- **Teacher-forced ≈ rolled-out** at every step — compounding errors aren't the issue;
  the single-step predictor itself produces minimal movement, so chained prediction stays close
  to the teacher-forced path

## Why the "copy_cos=1.0000" at later steps?

Late-trajectory states in the same file often converge to stable versions — the underlying
file content changes very little between commits after initial feature development. This
inflates the copy-last baseline artificially, making rollout comparisons harder to interpret
past step 2.

## Honest interpretation

The model learns a **weak local delta direction** (step 1 delta_cos ~0.10-0.16) that is
better than random, stronger for bigger capacity (ExpA > G10). It does not learn
**compositional transition dynamics** — multi-step rollouts are no better than copying
the previous state.

For a real world model, we'd need step 2+ rolled-out cosine to exceed copy-last with
reasonable margin. Neither model achieves this.
