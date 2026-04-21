"""Reusable LE-WM launcher bundle for Crucible external projects.

This launcher runs inside a cloned upstream ``lucas-maes/le-wm`` workspace.
It keeps the project spec thin while making the experiment entrypoint
shareable through local plugins, installed hub packages, or tap clones.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.cwd()))

# Import shared launcher utilities
_launchers_dir = str(Path(__file__).parent.parent)
if _launchers_dir not in sys.path:
    sys.path.insert(0, _launchers_dir)

from launchers._launcher_common import (
    apply_common_overrides,
    apply_slim_overrides,
    build_data_loaders,
    env_int,
    env_str,
    load_config as _load_config,
    make_jepa_forward,
    trim_encoder_depth,
    write_launch_metadata,
)


def run_training(cfg: Any, *, mode: str, slim: bool) -> None:
    import lightning as pl
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from lightning.pytorch.loggers import WandbLogger
    from omegaconf import OmegaConf, open_dict

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP, SIGReg
    from utils import ModelObjectCallBack

    common = apply_common_overrides(cfg, mode=mode)
    slim_details: dict[str, Any] = {}
    encoder_depth = None
    if slim:
        slim_details = apply_slim_overrides(cfg)
        encoder_depth = slim_details["encoder_depth"]

    batch_size = int(cfg.loader.batch_size)
    num_workers = int(cfg.loader.num_workers)

    train, val = build_data_loaders(cfg, batch_size=batch_size, num_workers=num_workers)

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    if slim and encoder_depth is not None:
        trim_encoder_depth(encoder, encoder_depth)

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    _forward = make_jepa_forward(cfg)

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=_forward,
        optim=optimizers,
    )

    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.subdir)
    write_launch_metadata(
        run_dir,
        {
            **common,
            **slim_details,
            "trainer_max_epochs": cfg.trainer.max_epochs,
            "loader_batch_size": cfg.loader.batch_size,
            "launcher": "lewm_upstream",
            "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        },
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )
    with open_dict(cfg):
        cfg.trainer.accelerator = "gpu"
        cfg.trainer.devices = 1

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )
    manager()


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    mode = env_str("LEWM_MODE", "slim")
    cfg = _load_config(args)
    run_training(cfg, mode=mode, slim=(mode == "slim"))


if __name__ == "__main__":
    main()
