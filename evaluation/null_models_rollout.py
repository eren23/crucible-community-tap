"""Delta-space null models for the multi-step compositional prediction claim.

The paper reports the per-step metric s_k = cos(pred_delta_k, true_delta_k)
where pred_delta_k is the predictor's latent displacement between rollout
step k-1 and step k, and true_delta_k is the ground-truth target-encoder
displacement.  The right nulls for this metric are delta-space nulls, not
state-space ones.

We compute four:

    trivial_identity     : zero-delta predictor (always 0 by convention)
    shuffled_action      : cos(true_delta_i, true_delta_{perm(i)})
                           for a random permutation within a batch
    random_trajectory    : cos(true_delta_i, true_delta_j) for i,j drawn
                           from different trajectories
    mean_delta           : cos(true_delta_i, mean(true_delta)) -- constant
                           predictor that always outputs the mean training
                           delta.  Tests whether the delta distribution is
                           low-rank enough that a mean-direction predictor
                           looks good by itself.

Plus one more delta-space null (self-consistency, cos(d01, d12)) that is
already in the script.

We also report the state-space measurements that explain WHY delta cosine
is the metric of choice.
"""
import sys, time
sys.path.insert(0, '/Users/eren/.crucible-hub/taps/crucible-community-tap')

import numpy as np
import torch
import h5py
from architectures.code_wm.code_wm import CodeWorldModel

CKPT = '/Users/eren/.crucible-hub/taps/crucible-community-tap/checkpoints/ema-frozen-15k-best.pt'
H5   = '/Users/eren/.crucible-hub/taps/crucible-community-tap/data/commitpack_python_trajectories_1.5m.h5'
HELD_OUT_FRACTION_START = 0.95
BATCH = 32
MAX_TRIPLES = 2000
SEED = 4242
DEVICE = 'cpu'

torch.manual_seed(SEED)
np.random.seed(SEED)

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg = ckpt['config']
model = CodeWorldModel(
    model_dim=cfg['model_dim'],
    num_loops=cfg['num_loops'],
    num_heads=cfg['num_heads'],
    encoder_loops=cfg['encoder_loops'],
    vocab_size=cfg['vocab_size'],
    max_seq_len=cfg['max_seq_len'],
    action_dim=cfg['action_dim'],
    ema_decay=cfg['ema_decay'],
)
model.load_state_dict(ckpt['model_state_dict'], strict=True)
model.train(False)
model.to(DEVICE)
print(f'[model] loaded step={ckpt["step"]} val_dcos_peak={ckpt["val_delta_cos"]:.4f}')


def encode(tokens_np):
    with torch.no_grad():
        x = torch.from_numpy(tokens_np.astype(np.int64)).to(DEVICE)
        z = model.target_encoder(x)
        if z.dim() == 3:
            z = z[:, 0, :]
    return z


with h5py.File(H5, 'r') as f:
    traj_offsets = f['trajectory/traj_offsets'][:]
    traj_lengths = f['trajectory/traj_lengths'][:]
    n_traj = len(traj_lengths)

    tail_start_traj = int(n_traj * HELD_OUT_FRACTION_START)
    held_out_ids = np.arange(tail_start_traj, n_traj)
    valid_ids = held_out_ids[traj_lengths[held_out_ids] >= 2]

    rng = np.random.default_rng(SEED)
    rng.shuffle(valid_ids)
    chosen = valid_ids[:MAX_TRIPLES]

    s0, s1, s2 = [], [], []
    for tid in chosen:
        off = int(traj_offsets[tid])
        s0.append(f['before_tokens'][off])
        s1.append(f['after_tokens'][off])
        s2.append(f['after_tokens'][off + 1])
    s0 = np.stack(s0, axis=0)
    s1 = np.stack(s1, axis=0)
    s2 = np.stack(s2, axis=0)
    print(f'[data] {s0.shape[0]} triples from held-out tail')


def batched_encode(tokens, batch=BATCH):
    out = []
    t0 = time.time()
    for i in range(0, tokens.shape[0], batch):
        out.append(encode(tokens[i:i + batch]))
    dt = time.time() - t0
    print(f'[encode] N={tokens.shape[0]} elapsed={dt:.1f}s')
    return torch.cat(out, dim=0)


def cos_rows(a, b):
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


print('[encode] s0...')
z0 = batched_encode(s0)
print('[encode] s1...')
z1 = batched_encode(s1)
print('[encode] s2...')
z2 = batched_encode(s2)

# ------- state-space measurements (for context) -----------------------
state_cb_01 = cos_rows(z0, z1)     # adjacent
state_skip  = cos_rows(z0, z2)     # skip-step

# ------- delta-space measurements -------------------------------------
# "true" deltas observed in data
d01 = z1 - z0
d02 = z2 - z0      # two-step state delta
d12 = z2 - z1      # next-step delta

# mean norms (sanity check)
print('\n[delta-space] norms (L2):')
print(f'  ||z0||      mean={z0.norm(dim=-1).mean():.3f}')
print(f'  ||d01||     mean={d01.norm(dim=-1).mean():.3f}')
print(f'  ||d02||     mean={d02.norm(dim=-1).mean():.3f}')
print(f'  ||d12||     mean={d12.norm(dim=-1).mean():.3f}')

# Null 1: shuffled-action null -- permute the true deltas within the batch and
# compute cosine between each true delta and a random other true delta
N = d01.shape[0]
perm = torch.randperm(N)
while (perm == torch.arange(N)).any():
    # Extremely unlikely to be identity but be safe
    perm = torch.randperm(N)
shuffled_null_d01 = cos_rows(d01, d01[perm])

# Null 2: random trajectory null for d02 (skip-step true delta)
perm2 = torch.randperm(N)
while (perm2 == torch.arange(N)).any():
    perm2 = torch.randperm(N)
random_traj_null_d02 = cos_rows(d02, d02[perm2])

# Null 3: self-consistency -- cos(d01, d12) within the SAME trajectory.
# If high, the delta direction is inherited from the trajectory itself
# rather than learned from the predictor.  This is the tightest null of
# all: "given the first delta, would predicting the same delta again
# work?"
self_consistency_d01_d12 = cos_rows(d01, d12)

# Null 4: mean-delta (constant predictor).
# What if you always predict the mean true delta?  Tests whether the
# delta distribution is low-rank enough that a constant mean-direction
# predictor scores high on its own.
mean_d01 = d01.mean(dim=0, keepdim=True)                # [1, D]
mean_delta_null = cos_rows(d01, mean_d01.expand_as(d01))


def summarise(name, t):
    t = t.float().cpu().numpy()
    print(f'  {name:36s}  mean={t.mean():.4f}  std={t.std():.4f}  '
          f'median={np.median(t):.4f}  n={len(t)}')


print('\n=== STATE-SPACE MEASUREMENTS (why we use delta cosine instead) ===')
summarise('state_cb  cos(z(s0), z(s1))', state_cb_01)
summarise('state_sk  cos(z(s0), z(s2))', state_skip)
print('\n  => Consecutive code states are already ~0.97 similar in absolute')
print('     state cosine, which is why we report delta cosine.')

print('\n=== DELTA-SPACE NULL MODELS (the right comparison for s1/s2/s3) ===')
summarise('trivial identity (always 0)', torch.zeros(1))
summarise('shuffled_null  cos(d01, d01[perm])', shuffled_null_d01)
summarise('random_traj_null  cos(d02, d02[perm])', random_traj_null_d02)
summarise('mean_delta_null  cos(d01, mean_d01)', mean_delta_null)
summarise('self_consistency  cos(d01, d12)',     self_consistency_d01_d12)

print('\nPaper headline to interpret:')
print('  predictor s1 (delta_cos_sim) at 15K champion ~ 0.987 - 0.991')
print('  trivial-identity null = 0.000')
print(f'  shuffled-action null (mean) = {shuffled_null_d01.float().mean():.4f}')
print(f'  random-trajectory null (mean) = {random_traj_null_d02.float().mean():.4f}')
print(f'  mean-delta (constant) null = {mean_delta_null.float().mean():.4f}')
print(f'  within-trajectory self-consistency null = {self_consistency_d01_d12.float().mean():.4f}')
