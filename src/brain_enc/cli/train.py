"""CLI entry point: train the brain encoder from a YAML config.

Usage
-----
    python -m brain_enc.cli.train --config configs/experiments/mirage.yaml

    # Override specific fields via dotted keys (hydra-style):
    python -m brain_enc.cli.train --config configs/experiments/mirage.yaml \\
        training.n_epochs=5 training.fast_dev_run=true
"""


import argparse
import json
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_OVERRIDE_PATTERN = re.compile(r"^[A-Za-z0-9_.]+=.+")


def _build_split_metadata(cfg, *, val_run_manifest=None) -> dict[str, object]:
    data_cfg = getattr(cfg, "data", None)
    custom_val_set = getattr(data_cfg, "custom_val_set", None)
    normalized_custom_val_selectors: list[str] = []
    if getattr(data_cfg, "split_strategy", None) == "custom_holdout":
        from brain_enc.data.algonauts import _normalize_custom_val_selectors

        normalized_custom_val_selectors = _normalize_custom_val_selectors(custom_val_set)

    payload: dict[str, object] = {
        "split_strategy": getattr(data_cfg, "split_strategy", None),
        "holdout_friends_season": getattr(data_cfg, "holdout_friends_season", None),
        "custom_val_set": custom_val_set,
        "resolved_custom_val_selectors": normalized_custom_val_selectors,
        "custom_val_name": getattr(data_cfg, "custom_val_name", None),
        "val_ratio": getattr(data_cfg, "val_ratio", None),
        "split_seed": getattr(data_cfg, "split_seed", None),
    }
    if val_run_manifest is not None and "stimulus_id" in val_run_manifest:
        payload["resolved_val_stimulus_ids"] = sorted(
            str(value) for value in val_run_manifest["stimulus_id"].dropna().unique().tolist()
        )
    if val_run_manifest is not None and "fmri_item_id" in val_run_manifest:
        payload["resolved_val_fmri_item_ids"] = sorted(
            str(value) for value in val_run_manifest["fmri_item_id"].dropna().unique().tolist()
        )
    return payload


def _split_train_cli_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split train CLI args into config paths and remaining argparse args."""

    config_args: list[str] = []
    remaining_args: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]

        if token == "--config":
            i += 1
            while i < len(argv):
                candidate = argv[i]
                if candidate.startswith("-") or _OVERRIDE_PATTERN.match(candidate):
                    break
                config_args.append(candidate)
                i += 1
            continue

        remaining_args.append(token)
        i += 1

    return config_args, remaining_args


def main(argv=None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    config_args, remaining_argv = _split_train_cli_args(raw_argv)

    parser = argparse.ArgumentParser(description="Train brain encoder from YAML config.")
    parser.add_argument(
        "--config",
        nargs="*",
        default=None,
        help="One or more YAML configs merged in order. Later configs override earlier ones.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Key=value overrides, e.g. training.n_epochs=5",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a fresh run in run_dir instead of resuming from last.ckpt.",
    )
    parser.add_argument(
        "--monitor",
        default=None,
        help="Validation metric to monitor for best.ckpt selection, e.g. val/pearson or val/loss.",
    )
    parser.add_argument(
        "--monitor-mode",
        choices=("min", "max"),
        default=None,
        help="Whether the monitored metric should be minimized or maximized for best.ckpt selection.",
    )
    parser.add_argument(
        "--val-set",
        default=None,
        help=(
            "Comma-separated custom validation selectors, e.g. "
            "'s6-18b,figures'. Supersedes data.split_strategy for this run."
        ),
    )
    parser.add_argument(
        "--val-name",
        default=None,
        help=(
            "Optional short name for the custom validation set, used in run "
            "metadata and auto-generated run names."
        ),
    )
    parser.add_argument(
        "--modality-stack-attn-dropout",
        type=float,
        default=None,
        help=(
            "Broadcast attn_dropout to modality_stack.{text,audio,vision}. "
            "Default None leaves each pooler's config value unchanged. "
            "Explicit per-modality overrides (e.g. modality_stack.text.attn_dropout=...) "
            "still win over this broadcast."
        ),
    )
    args = parser.parse_args(remaining_argv)
    args.config = config_args or args.config
    if not args.config:
        parser.error("the following arguments are required: --config")

    from brain_enc.cli._paths import resolve_config_paths
    from brain_enc.config_schema import load_config

    config_paths = resolve_config_paths(args.config)
    overrides: list[str] = []
    if args.modality_stack_attn_dropout is not None:
        for _modality in ("text", "audio", "vision"):
            overrides.append(
                f"modality_stack.{_modality}.attn_dropout={args.modality_stack_attn_dropout}"
            )
    overrides.extend(args.overrides)
    if args.val_set is not None:
        overrides.append("data.split_strategy=custom_holdout")
        overrides.append(f"data.custom_val_set={args.val_set}")
        if args.val_name is not None:
            overrides.append(f"data.custom_val_name={args.val_name}")

    logger.info("Loading config from %s", ", ".join(str(path) for path in config_paths))
    cfg = load_config(config_paths, overrides=overrides)
    if args.monitor is not None:
        cfg.training.monitor = args.monitor
    if args.monitor_mode is not None:
        cfg.training.monitor_mode = args.monitor_mode
    cfg.resolve_paths()

    logger.info("Run: %s  →  %s", cfg.run_name, cfg.run_dir)
    logger.info(
        "Checkpoint selection: monitor=%s mode=%s",
        cfg.training.monitor,
        cfg.training.monitor_mode,
    )

    # ----------------------------------------------------------------
    # Build data loaders from HDF5 feature stores
    # ----------------------------------------------------------------
    from brain_enc.data.construction import build_training_data_bundle
    from brain_enc.data.manifest_io import save_training_split_artifacts

    data_bundle = build_training_data_bundle(cfg)
    logger.info("Resolved data path: %s", data_bundle.resolved_datapath)
    feature_h5_paths = {
        modality: str(Path(store.path).resolve())
        for modality, store in data_bundle.feature_stores.items()
    }
    for modality, feature_h5_path in feature_h5_paths.items():
        logger.info("Feature store [%s]: %s", modality, feature_h5_path)
    fmri_h5_path = str(Path(data_bundle.fmri_store.path).resolve())
    logger.info("Feature store [fmri]: %s", fmri_h5_path)
    logger.info("Subjects: %d", data_bundle.n_subjects)
    logger.info(
        "Train batches: %d  |  Val batches: %d",
        len(data_bundle.train_loader),
        len(data_bundle.val_loader),
    )
    save_training_split_artifacts(
        cfg.run_dir,
        train_run_manifest=data_bundle.train_run_manifest,
        val_run_manifest=data_bundle.val_run_manifest,
        train_segment_manifest=data_bundle.train_segment_manifest,
        val_segment_manifest=data_bundle.val_segment_manifest,
    )
    with open(Path(cfg.run_dir) / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            _build_split_metadata(cfg, val_run_manifest=data_bundle.val_run_manifest),
            f,
            indent=2,
            sort_keys=True,
        )

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    from brain_enc.training.trainer import Experiment

    exp = Experiment(
        cfg,
        data_bundle.train_loader,
        data_bundle.val_loader,
        runtime_metadata={
            "resolved_data_path": str(data_bundle.resolved_datapath),
            "feature_h5_paths": feature_h5_paths,
            "fmri_h5_path": fmri_h5_path,
            "manifest_source": data_bundle.manifest_source,
            "manifest_bundle_dir": (
                str(data_bundle.manifest_bundle_dir.resolve())
                if data_bundle.manifest_bundle_dir is not None
                else ""
            ),
        },
        overwrite_run=args.overwrite,
    )
    metrics = exp.run()
    logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
