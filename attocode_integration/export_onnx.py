#!/usr/bin/env python3
"""Export Code WM checkpoint to ONNX for Synapse deployment.

Exports 3 separate ONNX models for the G8-style Code World Model:

1. encoder.onnx      -- byte tokens → 128-dim state embedding (for retrieval)
2. action_encoder.onnx -- 7-dim action → 128-dim action embedding
3. predictor.onnx    -- (z_state, z_action) → z_next (for prediction/rollout)

Synapse can load all three and compose them for full world-model inference,
or just encoder.onnx for retrieval-only features.

Usage::

    WM_POOL_MODE=cls python export_onnx.py \\
        --checkpoint checkpoint.pt \\
        --output-dir onnx_export/
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


def load_model(checkpoint_path: str, device: str = "cpu"):
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
    return model, cfg


# ---------------------------------------------------------------------------
# Wrappers: expose just the part we want to export
# ---------------------------------------------------------------------------

def _set_inference_mode(module):
    """Set module to inference mode and zero out any dropout."""
    module.train(False)
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


class EncoderWrapper(torch.nn.Module):
    """Wraps state_encoder — maps token IDs to a single embedding vector."""
    def __init__(self, model):
        super().__init__()
        self.encoder = model.state_encoder
        _set_inference_mode(self.encoder)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)  # [batch, model_dim]


class ActionEncoderWrapper(torch.nn.Module):
    """Wraps action_encoder — maps action vector to embedding."""
    def __init__(self, model):
        super().__init__()
        self.action_encoder = model.action_encoder
        _set_inference_mode(self.action_encoder)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(action)  # [batch, model_dim]


class PredictorWrapper(torch.nn.Module):
    """Wraps predictor — (z_state, z_action) → z_next."""
    def __init__(self, model):
        super().__init__()
        self.predictor = model.predictor
        _set_inference_mode(self.predictor)

    def forward(self, z_state: torch.Tensor, z_action: torch.Tensor) -> torch.Tensor:
        return self.predictor(z_state, z_action)  # [batch, model_dim]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(model, cfg, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    dim = cfg["model_dim"]
    seq = cfg["max_seq_len"]
    action_dim = cfg["action_dim"]

    # ---- Encoder ----
    print("Exporting encoder.onnx...")
    encoder = EncoderWrapper(model)
    dummy_tokens = torch.zeros(1, seq, dtype=torch.long)
    torch.onnx.export(
        encoder,
        (dummy_tokens,),
        str(output_dir / "encoder.onnx"),
        input_names=["tokens"],
        output_names=["state_embedding"],
        dynamic_axes={
            "tokens": {0: "batch"},
            "state_embedding": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"  → {output_dir / 'encoder.onnx'}")

    # ---- Action encoder ----
    print("Exporting action_encoder.onnx...")
    action_enc = ActionEncoderWrapper(model)
    dummy_action = torch.zeros(1, action_dim, dtype=torch.float32)
    torch.onnx.export(
        action_enc,
        (dummy_action,),
        str(output_dir / "action_encoder.onnx"),
        input_names=["action"],
        output_names=["action_embedding"],
        dynamic_axes={
            "action": {0: "batch"},
            "action_embedding": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"  → {output_dir / 'action_encoder.onnx'}")

    # ---- Predictor ----
    print("Exporting predictor.onnx...")
    predictor = PredictorWrapper(model)
    dummy_z_state = torch.zeros(1, dim, dtype=torch.float32)
    dummy_z_action = torch.zeros(1, dim, dtype=torch.float32)
    torch.onnx.export(
        predictor,
        (dummy_z_state, dummy_z_action),
        str(output_dir / "predictor.onnx"),
        input_names=["z_state", "z_action"],
        output_names=["z_next"],
        dynamic_axes={
            "z_state": {0: "batch"},
            "z_action": {0: "batch"},
            "z_next": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"  → {output_dir / 'predictor.onnx'}")

    # ---- Config ----
    runtime_config = {
        "model_name": "code_wm_g8",
        "model_dim": dim,
        "vocab_size": cfg["vocab_size"],
        "max_seq_len": seq,
        "action_dim": action_dim,
        "num_loops": cfg["num_loops"],
        "num_heads": cfg["num_heads"],
        "encoder_loops": cfg["encoder_loops"],
        "tokenizer": "python_ast",
        "pool_mode": "cls",
        "special_tokens": {
            "PAD": 612,
            "BOS": 613,
            "EOS": 614,
            "UNK": 615,
            "PARSE_ERROR": 616,
        },
        "recipe": "SIGReg (w=0.01) + delta-dir (w=0.5)",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(runtime_config, f, indent=2)
    print(f"  → {output_dir / 'config.json'}")


def verify_export(output_dir: Path, model, cfg):
    """Verify ONNX outputs match PyTorch outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping verification")
        return

    import numpy as np

    dim = cfg["model_dim"]
    seq = cfg["max_seq_len"]
    action_dim = cfg["action_dim"]

    # Test encoder
    print("\nVerifying encoder...")
    tokens_np = np.random.randint(0, cfg["vocab_size"], size=(2, seq), dtype=np.int64)
    with torch.no_grad():
        pt_out = model.state_encoder(torch.from_numpy(tokens_np)).numpy()
    sess = ort.InferenceSession(str(output_dir / "encoder.onnx"))
    onnx_out = sess.run(None, {"tokens": tokens_np})[0]
    max_diff = float(np.abs(pt_out - onnx_out).max())
    print(f"  Max diff: {max_diff:.2e} {'✓' if max_diff < 1e-4 else '⚠️'}")

    # Test predictor
    print("Verifying predictor...")
    zs = np.random.randn(2, dim).astype(np.float32)
    za = np.random.randn(2, dim).astype(np.float32)
    with torch.no_grad():
        pt_out = model.predictor(torch.from_numpy(zs), torch.from_numpy(za)).numpy()
    sess = ort.InferenceSession(str(output_dir / "predictor.onnx"))
    onnx_out = sess.run(None, {"z_state": zs, "z_action": za})[0]
    max_diff = float(np.abs(pt_out - onnx_out).max())
    print(f"  Max diff: {max_diff:.2e} {'✓' if max_diff < 1e-4 else '⚠️'}")


def main():
    parser = argparse.ArgumentParser(description="Export Code WM to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint}...")
    model, cfg = load_model(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, dim={cfg['model_dim']}\n")

    export_onnx(model, cfg, Path(args.output_dir))

    if not args.no_verify:
        verify_export(Path(args.output_dir), model, cfg)

    print(f"\nDone! Exported to {args.output_dir}")
    print("Ready for Synapse deployment.")


if __name__ == "__main__":
    main()
