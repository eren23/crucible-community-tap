#!/usr/bin/env python3
"""Baselines for the Code WM paper.

Answers the question: "Is CodeWM's 92-95% edit classification and 0.14
delta_cos actually impressive, or does a trivial method get close?"

Three baselines that work directly on the existing HDF5 data (no
retokenization needed):

1. Bag-of-AST-tokens: cosine similarity of token frequency vectors
2. TF-IDF + logistic regression: edit type classification
3. Random linear projection: random encoder → same eval pipeline

Usage::
    python baselines.py --data data/commitpackft_python_ast.h5 --num-samples 5000

Optionally compare against a trained CodeWM checkpoint::
    python baselines.py --data data/commitpackft_python_ast.h5 \
        --checkpoint checkpoints/code_wm_final.pt --num-samples 5000
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


# ---------------------------------------------------------------------------
# Data loading (reuse existing HDF5 format)
# ---------------------------------------------------------------------------

def load_hdf5_data(data_path: str, num_samples: int, seed: int = 42):
    """Load (before_tokens, after_tokens, edit_actions) from HDF5."""
    import h5py

    np.random.seed(seed)
    f = h5py.File(data_path, "r")
    n_total = f["before_tokens"].shape[0]
    n = min(num_samples, n_total)
    indices = np.sort(np.random.choice(n_total, size=n, replace=False))

    before = f["before_tokens"][indices.tolist()].astype(np.int64)
    after = f["after_tokens"][indices.tolist()].astype(np.int64)
    actions = f["edit_actions"][indices.tolist()].astype(np.float32)

    vocab_size = int(f["metadata"].attrs.get("vocab_size", 662))
    f.close()

    return before, after, actions, vocab_size


# ---------------------------------------------------------------------------
# Baseline 1: Bag-of-AST-tokens
# ---------------------------------------------------------------------------

def bag_of_tokens(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    """Convert token sequences to bag-of-tokens frequency vectors.

    Args:
        tokens: [N, seq_len] int64 token IDs
        vocab_size: vocabulary size

    Returns:
        [N, vocab_size] float32 normalized frequency vectors
    """
    N = tokens.shape[0]
    bows = np.zeros((N, vocab_size), dtype=np.float32)
    for i in range(N):
        for t in tokens[i]:
            if 0 <= t < vocab_size:
                bows[i, t] += 1.0
    # L2 normalize
    norms = np.linalg.norm(bows, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return bows / norms


def baseline_bag_of_tokens(
    before: np.ndarray,
    after: np.ndarray,
    actions: np.ndarray,
    vocab_size: int,
) -> dict:
    """Bag-of-tokens baseline: cosine sim of token frequency vectors."""
    print("\n=== Baseline 1: Bag-of-AST-tokens ===")
    t0 = time.time()

    bow_before = bag_of_tokens(before, vocab_size)
    bow_after = bag_of_tokens(after, vocab_size)

    # Delta in BoW space
    delta_bow = bow_after - bow_before
    delta_norms = np.linalg.norm(delta_bow, axis=1, keepdims=True)
    delta_norms = np.maximum(delta_norms, 1e-8)
    delta_bow_n = delta_bow / delta_norms

    # Cosine similarity between before and after (analogous to pred_cos)
    cos_ba = np.sum(bow_before * bow_after, axis=1).mean()

    # k-NN edit classification in BoW delta space
    edit_type = actions[:, :3].argmax(axis=1)
    scope = actions[:, 3:6].argmax(axis=1)
    joint_label = edit_type * 3 + scope

    n = len(joint_label)
    n_train = int(n * 0.8)
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    # k-NN with k=5
    train_d = delta_bow_n[train_idx]
    test_d = delta_bow_n[test_idx]
    sim = test_d @ train_d.T
    topk_idx = np.argsort(-sim, axis=1)[:, :5]

    train_labels = joint_label[train_idx]
    test_labels = joint_label[test_idx]
    neighbor_labels = train_labels[topk_idx]
    predictions = np.array([
        np.bincount(row, minlength=9).argmax() for row in neighbor_labels
    ])

    joint_acc = float((predictions == test_labels).mean())
    edit_acc = float(((predictions // 3) == (test_labels // 3)).mean())
    scope_acc = float(((predictions % 3) == (test_labels % 3)).mean())

    # Majority baseline
    unique, counts = np.unique(train_labels, return_counts=True)
    majority_acc = float((test_labels == unique[counts.argmax()]).mean())

    elapsed = time.time() - t0
    results = {
        "method": "bag_of_ast_tokens",
        "cos_before_after": float(cos_ba),
        "knn_joint_accuracy": joint_acc,
        "knn_edit_type_accuracy": edit_acc,
        "knn_scope_accuracy": scope_acc,
        "majority_baseline": majority_acc,
        "elapsed_s": elapsed,
    }

    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    return results


# ---------------------------------------------------------------------------
# Baseline 2: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

def baseline_tfidf_classifier(
    before: np.ndarray,
    after: np.ndarray,
    actions: np.ndarray,
    vocab_size: int,
) -> dict:
    """TF-IDF on token diffs → logistic regression for edit classification."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("\n=== Baseline 2: TF-IDF + Logistic Regression ===")
    t0 = time.time()

    bow_before = bag_of_tokens(before, vocab_size)
    bow_after = bag_of_tokens(after, vocab_size)

    # Feature: concatenation of before BoW + delta BoW
    delta_bow = bow_after - bow_before
    features = np.concatenate([bow_before, delta_bow], axis=1)

    # Labels
    edit_type = actions[:, :3].argmax(axis=1)
    scope = actions[:, 3:6].argmax(axis=1)
    joint_label = edit_type * 3 + scope

    n = len(joint_label)
    n_train = int(n * 0.8)
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    # Edit type classification
    clf_edit = LogisticRegression(max_iter=1000, random_state=42)
    clf_edit.fit(features[train_idx], edit_type[train_idx])
    edit_acc = accuracy_score(edit_type[test_idx], clf_edit.predict(features[test_idx]))

    # Scope classification
    clf_scope = LogisticRegression(max_iter=1000, random_state=42)
    clf_scope.fit(features[train_idx], scope[train_idx])
    scope_acc = accuracy_score(scope[test_idx], clf_scope.predict(features[test_idx]))

    # Joint classification
    clf_joint = LogisticRegression(max_iter=1000, random_state=42)
    clf_joint.fit(features[train_idx], joint_label[train_idx])
    joint_pred = clf_joint.predict(features[test_idx])
    joint_acc = accuracy_score(joint_label[test_idx], joint_pred)

    # Majority baseline
    unique, counts = np.unique(joint_label[train_idx], return_counts=True)
    majority_acc = float((joint_label[test_idx] == unique[counts.argmax()]).mean())

    elapsed = time.time() - t0
    results = {
        "method": "tfidf_logreg",
        "edit_type_accuracy": float(edit_acc),
        "scope_accuracy": float(scope_acc),
        "joint_accuracy": float(joint_acc),
        "majority_baseline": float(majority_acc),
        "lift_over_majority": float(joint_acc - majority_acc),
        "elapsed_s": elapsed,
    }

    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    return results


# ---------------------------------------------------------------------------
# Baseline 3: Random Linear Projection
# ---------------------------------------------------------------------------

def baseline_random_projection(
    before: np.ndarray,
    after: np.ndarray,
    actions: np.ndarray,
    vocab_size: int,
    latent_dim: int = 128,
) -> dict:
    """Random linear encoder: tokens → random projection → same eval metrics."""
    print(f"\n=== Baseline 3: Random Linear Projection (dim={latent_dim}) ===")
    t0 = time.time()

    torch.manual_seed(42)

    # Random projection matrix (fixed, not learned)
    proj = torch.randn(vocab_size, latent_dim) / (latent_dim ** 0.5)

    bow_before = torch.from_numpy(bag_of_tokens(before, vocab_size))
    bow_after = torch.from_numpy(bag_of_tokens(after, vocab_size))

    # Project to latent space
    z_before = bow_before @ proj  # [N, D]
    z_after = bow_after @ proj

    # Delta in projected space
    delta_true = z_after - z_before
    # "Predicted" delta = just the true delta (random encoder has no predictor)
    # So we measure: can random geometry support the same downstream tasks?

    # Cosine sim before-after
    cos_ba = F.cosine_similarity(z_before, z_after, dim=-1).mean().item()

    # k-NN classification on delta vectors
    edit_type = actions[:, :3].argmax(axis=1)
    scope = actions[:, 3:6].argmax(axis=1)
    joint_label = edit_type * 3 + scope

    n = len(joint_label)
    n_train = int(n * 0.8)
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    delta_n = F.normalize(delta_true, dim=-1)
    train_d = delta_n[train_idx]
    test_d = delta_n[test_idx]
    sim = test_d @ train_d.T
    _, topk_idx = sim.topk(5, dim=-1)

    train_labels = joint_label[train_idx]
    test_labels = joint_label[test_idx]
    neighbor_labels = train_labels[topk_idx.numpy()]
    predictions = np.array([
        np.bincount(row, minlength=9).argmax() for row in neighbor_labels
    ])

    joint_acc = float((predictions == test_labels).mean())
    edit_acc = float(((predictions // 3) == (test_labels // 3)).mean())

    # Majority baseline
    unique, counts = np.unique(train_labels, return_counts=True)
    majority_acc = float((test_labels == unique[counts.argmax()]).mean())

    # Retrieval: k-NN in delta space, measure action similarity
    actions_t = torch.from_numpy(actions).float()
    actions_n = F.normalize(actions_t, dim=-1)
    sim_matrix = delta_n @ delta_n.T
    sim_matrix.fill_diagonal_(-float("inf"))
    _, topk_ret = sim_matrix.topk(10, dim=-1)
    neighbor_actions = actions_n[topk_ret]
    query_actions = actions_n.unsqueeze(1)
    action_sim = (query_actions * neighbor_actions).sum(dim=-1).mean().item()
    random_idx = torch.randint(0, n, (n, 10))
    random_sim = (query_actions * actions_n[random_idx]).sum(dim=-1).mean().item()

    elapsed = time.time() - t0
    results = {
        "method": "random_projection",
        "latent_dim": latent_dim,
        "cos_before_after": float(cos_ba),
        "knn_joint_accuracy": joint_acc,
        "knn_edit_type_accuracy": edit_acc,
        "majority_baseline": float(majority_acc),
        "retrieval_action_sim": action_sim,
        "retrieval_random_sim": random_sim,
        "retrieval_lift": action_sim - random_sim,
        "elapsed_s": elapsed,
    }

    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baselines for Code WM paper")
    parser.add_argument("--data", required=True, help="HDF5 data file")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading data from {args.data}...")
    before, after, actions, vocab_size = load_hdf5_data(
        args.data, args.num_samples, args.seed
    )
    print(f"  {before.shape[0]} samples, vocab_size={vocab_size}")

    # Run all baselines
    r1 = baseline_bag_of_tokens(before, after, actions, vocab_size)
    r2 = baseline_tfidf_classifier(before, after, actions, vocab_size)
    r3 = baseline_random_projection(before, after, actions, vocab_size, args.latent_dim)

    # Summary comparison table
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Edit Type Acc':<15} {'Joint Acc':<15} {'Retrieval Lift':<15}")
    print("-" * 70)
    print(f"{'Majority class':<30} {'--':<15} {r1['majority_baseline']:<15.3f} {'--':<15}")
    print(f"{'Bag-of-AST-tokens (k-NN)':<30} {r1['knn_edit_type_accuracy']:<15.3f} {r1['knn_joint_accuracy']:<15.3f} {'--':<15}")
    print(f"{'TF-IDF + LogReg':<30} {r2['edit_type_accuracy']:<15.3f} {r2['joint_accuracy']:<15.3f} {'--':<15}")
    print(f"{'Random projection (k-NN)':<30} {r3['knn_edit_type_accuracy']:<15.3f} {r3['knn_joint_accuracy']:<15.3f} {r3['retrieval_lift']:<15.3f}")
    print("-" * 70)
    print("Compare against CodeWM: edit_type ~92-95%, joint ~71-78%, retrieval lift ~0.25-0.29")
    print()


if __name__ == "__main__":
    main()
