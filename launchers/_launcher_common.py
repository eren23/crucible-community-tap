"""Shared launcher utilities for LE-WM family launchers.

Provides env-var parsing helpers, config loading, Hydra override application,
and metadata writing that were copy-pasted across lewm_upstream,
hybrid_lewm_upstream, and elastic_lewm_upstream.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "")
    return raw or default


# ---------------------------------------------------------------------------
# Model attribute helpers
# ---------------------------------------------------------------------------

def set_nested_attr(obj: Any, dotted_path: str, value: Any) -> bool:
    current = obj
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if not hasattr(current, part):
            return False
        current = getattr(current, part)
    if not hasattr(current, parts[-1]):
        return False
    setattr(current, parts[-1], value)
    return True


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------

def resolve_run_identity(default_name: str) -> tuple[str, str]:
    """Resolve run name and ID from Crucible/legacy env vars."""
    run_name = env_str(
        "CRUCIBLE_VARIANT_NAME",
        env_str("LEWM_VARIANT", env_str("WANDB_RUN_NAME", default_name)),
    )
    run_id = env_str("CRUCIBLE_RUN_ID", env_str("WANDB_RUN_NAME", run_name))
    return run_name, run_id


# ---------------------------------------------------------------------------
# Encoder trimming
# ---------------------------------------------------------------------------

def trim_encoder_depth(encoder: Any, depth: int) -> None:
    import torch.nn as nn

    candidate_paths = (
        "encoder.layer",
        "vit.encoder.layer",
        "blocks",
        "layers",
    )

    for path in candidate_paths:
        current = encoder
        found = True
        for part in path.split("."):
            if not hasattr(current, part):
                found = False
                break
            current = getattr(current, part)
        if not found or not isinstance(current, (list, nn.ModuleList)):
            continue
        if len(current) < depth:
            raise ValueError(f"Requested encoder depth {depth}, but {path} only has {len(current)} layers")
        trimmed = type(current)(list(current)[:depth])
        if set_nested_attr(encoder, path, trimmed):
            if hasattr(encoder, "config") and hasattr(encoder.config, "num_hidden_layers"):
                encoder.config.num_hidden_layers = depth
            return
    raise ValueError("Could not find a trimmable encoder layer stack on the upstream LE-WM encoder")


# ---------------------------------------------------------------------------
# Hydra config
# ---------------------------------------------------------------------------

def load_config(dotlist: list[str]) -> Any:
    from hydra import compose, initialize_config_dir

    config_dir = Path.cwd() / "config" / "train"
    if not config_dir.exists():
        raise FileNotFoundError(f"Expected upstream LE-WM config directory at {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="lewm", overrides=dotlist)


# ---------------------------------------------------------------------------
# Common overrides
# ---------------------------------------------------------------------------

def apply_common_overrides(cfg: Any, *, mode: str) -> dict[str, Any]:
    from omegaconf import open_dict

    run_name, run_id = resolve_run_identity(str(getattr(cfg, "output_model_name", "lewm")))
    num_workers = env_int("SLIM_NUM_WORKERS", int(getattr(cfg, "num_workers", 4)))
    batch_size = env_int("SLIM_BATCH_SIZE", int(getattr(cfg.loader, "batch_size", 128)))
    max_epochs = env_int("SLIM_MAX_EPOCHS", int(getattr(cfg.trainer, "max_epochs", 100)))
    sigreg_num_proj = env_int("SLIM_SIGREG_PROJ", int(cfg.loss.sigreg.kwargs.get("num_proj", 1024)))

    with open_dict(cfg):
        cfg.output_model_name = run_name
        cfg.subdir = run_id
        cfg.num_workers = num_workers
        cfg.loader.num_workers = num_workers
        cfg.loader.batch_size = batch_size
        cfg.loader.persistent_workers = num_workers > 0
        if hasattr(cfg.loader, "prefetch_factor") and num_workers <= 0:
            cfg.loader.prefetch_factor = None
        cfg.trainer.max_epochs = max_epochs
        cfg.loss.sigreg.kwargs.num_proj = sigreg_num_proj
        if getattr(cfg, "wandb", None) and getattr(cfg.wandb, "config", None):
            cfg.wandb.config.name = run_name
            cfg.wandb.config.id = run_id
            cfg.wandb.config.entity = env_str("WANDB_ENTITY", str(getattr(cfg.wandb.config, "entity", "") or ""))
            cfg.wandb.config.project = env_str("WANDB_PROJECT", str(getattr(cfg.wandb.config, "project", "") or ""))
            cfg.wandb.enabled = os.environ.get("WANDB_MODE", "online") != "disabled"

    return {
        "mode": mode,
        "run_name": run_name,
        "run_id": run_id,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "num_workers": num_workers,
        "sigreg_num_proj": sigreg_num_proj,
    }


def apply_slim_overrides(cfg: Any) -> dict[str, Any]:
    from omegaconf import open_dict

    encoder_depth = env_int("SLIM_ENC_DEPTH", 0)
    predictor_depth = env_int("SLIM_PRED_DEPTH", int(getattr(cfg.predictor, "depth", 6)))
    embed_dim = env_int("SLIM_DIM", int(getattr(cfg.wm, "embed_dim", 192)))
    if encoder_depth <= 0:
        raise ValueError("SLIM_ENC_DEPTH must be set to a positive integer for slim runs")

    with open_dict(cfg):
        cfg.wm.embed_dim = embed_dim
        cfg.predictor.depth = predictor_depth

    return {
        "embed_dim": embed_dim,
        "encoder_depth": encoder_depth,
        "predictor_depth": predictor_depth,
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_launch_metadata(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "crucible_launch_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Data pipeline (shared HF + HDF5 setup)
# ---------------------------------------------------------------------------

def build_data_loaders(cfg: Any, *, batch_size: int, num_workers: int):
    """Build train and val DataLoaders from precomputed HF or raw HDF5 data.

    Returns (train_loader, val_loader).
    """
    import os
    import torch
    from omegaconf import open_dict

    precomputed = os.environ.get("PRECOMPUTED_DATASET", "")
    if precomputed:
        from datasets import load_dataset as _load_hf
        _hf_token = os.environ.get("HF_TOKEN") or None
        _hf_ds = _load_hf(precomputed, token=_hf_token)
        _hf_ds.set_format("torch")
        train_set = _hf_ds["train"]
        val_set = _hf_ds["validation"] if "validation" in _hf_ds else _hf_ds["test"]

        with open_dict(cfg):
            frameskip = int(cfg.data.dataset.get("frameskip", 1))
            for col in cfg.data.dataset.keys_to_load:
                if col.startswith("pixels"):
                    continue
                sample = train_set[0]
                if col in sample:
                    raw_dim = sample[col].shape[-1] if sample[col].dim() > 1 else 1
                    if col == "action" and frameskip > 1:
                        raw_dim = raw_dim // frameskip
                    setattr(cfg.wm, f"{col}_dim", raw_dim)

        train = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        val = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    else:
        import stable_pretraining as spt
        import stable_worldmodel as swm
        from utils import get_column_normalizer, get_img_preprocessor

        dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
        transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

        with open_dict(cfg):
            for col in cfg.data.dataset.keys_to_load:
                if col.startswith("pixels"):
                    continue
                normalizer = get_column_normalizer(dataset, col, col)
                transforms.append(normalizer)
                setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

        transform = spt.data.transforms.Compose(*transforms)
        dataset.transform = transform

        rnd_gen = torch.Generator().manual_seed(cfg.seed)
        train_set, val_set = spt.data.random_split(
            dataset,
            lengths=[cfg.train_split, 1 - cfg.train_split],
            generator=rnd_gen,
        )

        train = torch.utils.data.DataLoader(
            train_set, **cfg.loader, shuffle=True, drop_last=True,
            generator=rnd_gen,
        )
        val = torch.utils.data.DataLoader(
            val_set, **cfg.loader, shuffle=False, drop_last=False,
        )

    return train, val


# ---------------------------------------------------------------------------
# Forward function (shared JEPA loss computation)
# ---------------------------------------------------------------------------

def make_jepa_forward(cfg: Any):
    """Create the standard JEPA forward function used by all LE-WM launchers.

    Returns a function compatible with spt.Module(forward=...).
    """
    import torch

    def _forward(self, batch, stage):
        ctx_len = cfg.wm.history_size
        n_preds = cfg.wm.num_preds
        lambd = cfg.loss.sigreg.weight

        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        output = self.model.encode(batch)
        emb = output["emb"]
        act_emb = output["act_emb"]
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]
        pred_emb = self.model.predict(ctx_emb, ctx_act)
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
        losses = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
        is_val = stage == "validate"
        self.log_dict(
            losses,
            on_step=not is_val,
            on_epoch=is_val,
            sync_dist=True,
        )
        return output

    return _forward
