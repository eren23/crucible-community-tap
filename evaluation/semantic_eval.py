#!/usr/bin/env python3
"""Semantic downstream tests for Code World Model.

Tests whether the learned latent geometry is actually useful, beyond
val/cosine_sim. Four probes:

1. Edit Retrieval: For each query edit, find k-NN in delta space.
   Measure action-vector similarity between query and retrieved neighbors.

2. k-NN Edit Classification: Use delta vectors as features to predict
   (edit_type, scope) labels via k-NN.

3. Cluster Purity: k-means cluster delta vectors, check alignment
   with ground-truth (edit_type x scope) labels.

4. Held-out prediction quality on unseen data.

Usage::
    python semantic_eval.py --checkpoint checkpoints/code_wm_final.pt \\
        --data data/commitpackft_python_ast.h5 --num-samples 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def load_model_and_data(
    checkpoint_path: str,
    data_path: str,
    num_samples: int,
    device: str,
):
    """Load model from checkpoint plus held-out sample data."""
    import h5py
    import importlib.util

    tap_root = Path(__file__).parent.parent
    for mod_name, mod_path in [
        ("wm_base", tap_root / "architectures" / "wm_base" / "wm_base.py"),
        ("code_wm", tap_root / "architectures" / "code_wm" / "code_wm.py"),
    ]:
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

    import code_wm

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = code_wm.CodeWorldModel(
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_seq_len"],
        encoder_loops=cfg["encoder_loops"],
        model_dim=cfg["model_dim"],
        num_loops=cfg["num_loops"],
        num_heads=cfg["num_heads"],
        predictor_depth=2,
        ema_decay=cfg["ema_decay"],
        action_dim=cfg["action_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.train(False)

    f = h5py.File(data_path, "r")
    n_total = f["before_tokens"].shape[0]
    n = min(num_samples, n_total)
    indices = np.sort(np.random.choice(n_total, size=n, replace=False))

    before = torch.from_numpy(f["before_tokens"][indices.tolist()].astype(np.int64)).to(device)
    actions = torch.from_numpy(f["edit_actions"][indices.tolist()].astype(np.float32)).to(device)
    after = torch.from_numpy(f["after_tokens"][indices.tolist()].astype(np.int64)).to(device)
    f.close()

    return model, before, actions, after, cfg


@torch.no_grad()
def compute_embeddings_and_deltas(
    model: Any,
    before: torch.Tensor,
    actions: torch.Tensor,
    after: torch.Tensor,
    batch_size: int = 64,
):
    """Run model, return raw embeddings and deltas."""
    n = before.shape[0]
    z_before_list, z_after_list, delta_pred_list = [], [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        z_b = model.state_encoder(before[start:end])
        z_a = model.target_encoder(after[start:end])
        z_act = model.action_encoder(actions[start:end])
        z_pred = model.predictor(z_b, z_act)

        z_before_list.append(z_b.cpu())
        z_after_list.append(z_a.cpu())
        delta_pred_list.append((z_pred - z_b).cpu())

    z_before = torch.cat(z_before_list)
    z_after = torch.cat(z_after_list)
    delta_pred = torch.cat(delta_pred_list)
    delta_true = z_after - z_before

    return z_before, z_after, delta_pred, delta_true


def test_edit_retrieval(
    delta_true: torch.Tensor,
    actions: np.ndarray,
    k: int = 10,
) -> dict:
    """For each edit, find k-NN in delta space. Higher action similarity = good."""
    delta_n = F.normalize(delta_true, dim=-1)
    sim_matrix = delta_n @ delta_n.T
    sim_matrix.fill_diagonal_(-float("inf"))
    _, topk_idx = sim_matrix.topk(k, dim=-1)

    actions_t = torch.from_numpy(actions).float()
    actions_n = F.normalize(actions_t, dim=-1)

    neighbor_actions = actions_n[topk_idx]
    query_actions = actions_n.unsqueeze(1)
    action_sim = (query_actions * neighbor_actions).sum(dim=-1)
    mean_action_sim = action_sim.mean().item()

    random_idx = torch.randint(0, len(actions), (len(actions), k))
    random_neighbor_actions = actions_n[random_idx]
    random_sim = (query_actions * random_neighbor_actions).sum(dim=-1).mean().item()

    return {
        "delta_nn_action_sim": mean_action_sim,
        "random_nn_action_sim": random_sim,
        "lift": mean_action_sim - random_sim,
    }


def test_knn_classification(
    delta_true: torch.Tensor,
    actions: np.ndarray,
    k: int = 5,
    train_frac: float = 0.8,
) -> dict:
    """k-NN classify (edit_type, scope) from delta vectors."""
    edit_type = actions[:, :3].argmax(axis=1)
    scope = actions[:, 3:6].argmax(axis=1)
    joint_label = edit_type * 3 + scope

    n = len(joint_label)
    n_train = int(n * train_frac)
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    train_delta = F.normalize(delta_true[train_idx], dim=-1)
    test_delta = F.normalize(delta_true[test_idx], dim=-1)
    train_labels = joint_label[train_idx]
    test_labels = joint_label[test_idx]

    sim = test_delta @ train_delta.T
    _, topk_idx = sim.topk(k, dim=-1)
    neighbor_labels = train_labels[topk_idx.numpy()]

    # Majority vote per row
    predictions = np.array([
        np.bincount(row, minlength=9).argmax() for row in neighbor_labels
    ])
    accuracy = (predictions == test_labels).mean()

    unique, counts = np.unique(train_labels, return_counts=True)
    majority_class = unique[counts.argmax()]
    majority_acc = (test_labels == majority_class).mean()

    pred_edit = predictions // 3
    pred_scope = predictions % 3
    true_edit = test_labels // 3
    true_scope = test_labels % 3

    return {
        "knn_joint_accuracy": float(accuracy),
        "knn_edit_type_accuracy": float((pred_edit == true_edit).mean()),
        "knn_scope_accuracy": float((pred_scope == true_scope).mean()),
        "majority_baseline": float(majority_acc),
        "lift": float(accuracy - majority_acc),
    }


def test_cluster_purity(
    delta_true: torch.Tensor,
    actions: np.ndarray,
) -> dict:
    """k-means on delta vectors, measure label alignment."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    edit_type = actions[:, :3].argmax(axis=1)
    scope = actions[:, 3:6].argmax(axis=1)
    joint_label = edit_type * 3 + scope

    delta_n = F.normalize(delta_true, dim=-1).numpy()

    results = {}
    for k, label_name, labels in [
        (3, "edit_type", edit_type),
        (3, "scope", scope),
        (9, "joint", joint_label),
    ]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = km.fit_predict(delta_n)

        nmi = normalized_mutual_info_score(labels, clusters)
        ari = adjusted_rand_score(labels, clusters)
        results[f"cluster_{label_name}_nmi"] = float(nmi)
        results[f"cluster_{label_name}_ari"] = float(ari)

    return results


def test_prediction_quality(
    z_before: torch.Tensor,
    z_after: torch.Tensor,
    delta_pred: torch.Tensor,
    delta_true: torch.Tensor,
) -> dict:
    """Direct prediction quality on held-out data."""
    z_pred = z_before + delta_pred

    cos_pred = F.cosine_similarity(z_pred, z_after, dim=-1).mean().item()
    dt_n = F.normalize(delta_true, dim=-1)
    dp_n = F.normalize(delta_pred, dim=-1)
    delta_cos = (dt_n * dp_n).sum(dim=-1).mean().item()

    dt_norm = delta_true.norm(dim=-1).clamp_min(1e-6)
    dp_norm = delta_pred.norm(dim=-1).clamp_min(1e-6)
    norm_ratio = (dp_norm / dt_norm).mean().item()

    cos_before_after = F.cosine_similarity(z_before, z_after, dim=-1).mean().item()
    random_pred = torch.randn_like(z_pred)
    cos_random = F.cosine_similarity(random_pred, z_after, dim=-1).mean().item()

    return {
        "heldout_cos_pred_vs_after": cos_pred,
        "heldout_delta_cos_sim": delta_cos,
        "heldout_delta_norm_ratio": norm_ratio,
        "baseline_cos_before_vs_after": cos_before_after,
        "baseline_cos_random_vs_after": cos_random,
    }


def main():
    parser = argparse.ArgumentParser(description="Semantic tests for Code WM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model from {args.checkpoint}...")
    model, before, actions_t, after, cfg = load_model_and_data(
        args.checkpoint, args.data, args.num_samples, args.device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, dim={cfg['model_dim']}, vocab={cfg['vocab_size']}")
    print(f"Data: {before.shape[0]} held-out samples")

    print("\nComputing embeddings + deltas...")
    t0 = time.time()
    z_before, z_after, delta_pred, delta_true = compute_embeddings_and_deltas(
        model, before, actions_t, after,
    )
    print(f"  done in {time.time()-t0:.1f}s")

    actions_np = actions_t.cpu().numpy()

    print("\n=== Test 4: Held-out prediction quality ===")
    r4 = test_prediction_quality(z_before, z_after, delta_pred, delta_true)
    for k, v in r4.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Test 1: Edit retrieval (k-NN in delta space) ===")
    r1 = test_edit_retrieval(delta_true, actions_np, k=10)
    for k, v in r1.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Test 2: k-NN edit classification ===")
    r2 = test_knn_classification(delta_true, actions_np, k=5)
    for k, v in r2.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Test 3: Cluster purity ===")
    r3 = test_cluster_purity(delta_true, actions_np)
    for k, v in r3.items():
        print(f"  {k}: {v:.4f}")

    # Summary verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    checks = []
    checks.append(("Delta direction alignment", r4["heldout_delta_cos_sim"], 0.3))
    checks.append(("Delta NN retrieval lift", r1["lift"], 0.1))
    checks.append(("k-NN classification lift", r2["lift"], 0.05))
    checks.append(("Cluster NMI (joint)", r3["cluster_joint_nmi"], 0.05))

    for name, val, threshold in checks:
        mark = "[PASS]" if val > threshold else "[FAIL]"
        print(f"  {mark} {name}: {val:.3f} (threshold: {threshold})")


if __name__ == "__main__":
    main()
