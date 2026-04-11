"""Shared CodeWM eval helpers.

Canonical implementations for the duplicated functions that used to live
in every single `evaluation/code_wm/*.py` script. Pulled out into one
module so a bug fix (e.g. the 2026-04-10 ``WM_POOL_MODE=cls``/
``strict=False`` silent-drop bug) only needs to be made once.

Public API:

- ``load_codewm(checkpoint_path, device="cpu") -> (model, config_dict)``
  Canonical checkpoint loader. Forces ``WM_POOL_MODE=attn`` to match training.
- ``load_model``: backward-compat alias for ``load_codewm`` (some scripts
  historically named it this way).
- ``resolve_tap_root() -> Path``: idempotently resolves the tap root so
  ``from collectors.ast_tokenizer import ast_tokenize`` and friends work
  regardless of cwd / pod layout.

Usage in a script::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))  # let us import _shared
    from _shared import load_codewm, resolve_tap_root

    model, cfg = load_codewm("/path/to/checkpoint.pt", device="cpu")
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import torch

__all__ = ["load_codewm", "load_model", "resolve_tap_root"]


# Module-level cache so the (expensive) dynamic imports of ``wm_base`` and
# ``code_wm`` only happen once per process even when multiple eval scripts
# import this module.
_CODE_WM_CACHE: tuple[Any, Path] | None = None


def resolve_tap_root() -> Path:
    """Resolve the tap root directory.

    Tries the file-relative path first (for local runs where the script
    sits inside the tap clone), then falls back to ``/workspace/
    crucible-community-tap`` (the RunPod pod layout where the tap is
    rsynced during bootstrap).
    """
    here = Path(__file__).parent.parent.parent
    if (here / "architectures" / "wm_base" / "wm_base.py").exists():
        return here
    return Path("/workspace/crucible-community-tap")


def _ensure_modules_loaded() -> tuple[Any, Path]:
    """Idempotently import the tap's ``wm_base`` and ``code_wm`` modules.

    Also prepends the tap root to ``sys.path`` so sibling-package imports
    like ``from collectors.ast_tokenizer import ast_tokenize`` work.
    """
    global _CODE_WM_CACHE
    if _CODE_WM_CACHE is not None:
        return _CODE_WM_CACHE

    tap_root = resolve_tap_root()
    tap_root_str = str(tap_root)
    if tap_root_str not in sys.path:
        sys.path.insert(0, tap_root_str)

    for mod_name, mod_path in [
        ("wm_base", tap_root / "architectures" / "wm_base" / "wm_base.py"),
        ("code_wm", tap_root / "architectures" / "code_wm" / "code_wm.py"),
    ]:
        if mod_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {mod_name} from {mod_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)

    import code_wm  # type: ignore  # noqa: E402
    _CODE_WM_CACHE = (code_wm, tap_root)
    return _CODE_WM_CACHE


def load_codewm(checkpoint_path: str, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """Load a CodeWM checkpoint and return ``(model, config_dict)``.

    Forces ``WM_POOL_MODE=attn`` to match training (the default). This
    prevents the 2026-04-10 silent bug where forcing ``cls`` at eval
    time built the model without the ``attn_pool`` module, causing
    ``model.load_state_dict(strict=False)`` to silently drop the
    ``attn_pool.*`` weights and return an untrained readout. Do not
    override this unless you explicitly understand what you're doing
    (and your checkpoint was trained with ``WM_POOL_MODE=cls``).
    """
    os.environ.setdefault("WM_POOL_MODE", "attn")
    code_wm, _ = _ensure_modules_loaded()
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
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"  [warn] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected:
            print(f"    unexpected sample: {unexpected[:3]}")
    model.to(device)
    model.train(False)
    return model, cfg


# Backward-compat alias. Some older scripts use ``load_model(...)`` instead
# of ``load_codewm(...)``. They are functionally identical; the two names
# are maintained to avoid a flood of tiny find-and-replace commits.
load_model = load_codewm
