"""CLI: build a canonical TSV manifest bundle from the raw dataset tree."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import typing as tp

from brain_enc.cli.train import _split_train_cli_args

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(argv: tp.Sequence[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    config_args, remaining_argv = _split_train_cli_args(raw_argv)

    parser = argparse.ArgumentParser(
        description="Build a canonical TSV manifest bundle from the raw dataset tree."
    )
    parser.add_argument(
        "--config",
        nargs="*",
        default=None,
        help="One or more YAML configs merged in order. Later configs override earlier ones.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional key=value overrides, e.g. data.datapath=/tmp/dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing manifest bundle directory.",
    )
    args = parser.parse_args(remaining_argv)
    args.config = config_args or args.config
    if not args.config:
        parser.error("the following arguments are required: --config")

    from brain_enc.cli._paths import resolve_config_paths
    from brain_enc.config_schema import load_config
    from brain_enc.data.algonauts import build_manifest, build_ood_manifest, build_s7_manifest
    from brain_enc.data.manifest_io import (
        build_manifest_bundle_id,
        default_manifest_dir,
        prepare_canonical_benchmark_manifest,
        prepare_canonical_run_manifest,
        prepare_canonical_stimulus_manifest,
        save_manifest_bundle,
    )
    from brain_enc.env import get_datapath

    config_paths = resolve_config_paths(args.config)
    cfg = load_config(config_paths, overrides=args.overrides)
    cfg.resolve_paths()

    datapath = Path(cfg.data.datapath) if cfg.data.datapath else get_datapath()
    resolved_datapath = datapath.resolve()
    bundle_dir = default_manifest_dir(cfg)
    if bundle_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Manifest bundle already exists at {bundle_dir}. Use --overwrite to replace it."
        )

    logger.info("Scanning raw dataset at %s", resolved_datapath)
    raw_run_manifest = build_manifest(resolved_datapath)
    raw_s7_manifest = build_s7_manifest(resolved_datapath)
    raw_ood_manifest = build_ood_manifest(resolved_datapath)

    run_manifest = prepare_canonical_run_manifest(
        raw_run_manifest,
        dataset_name=cfg.data.dataset_name,
        datapath=resolved_datapath,
    )
    stimulus_manifest = prepare_canonical_stimulus_manifest(
        run_manifest,
        dataset_name=cfg.data.dataset_name,
        datapath=resolved_datapath,
    )
    friends_s7_manifest = prepare_canonical_benchmark_manifest(
        raw_s7_manifest,
        dataset_name=cfg.data.dataset_name,
        datapath=resolved_datapath,
    )
    ood_manifest = prepare_canonical_benchmark_manifest(
        raw_ood_manifest,
        dataset_name=cfg.data.dataset_name,
        datapath=resolved_datapath,
    )

    metadata = {
        "manifest_schema_version": "v1",
        "dataset_name": cfg.data.dataset_name,
        "stimulus_namespace": cfg.data.dataset_name,
        "bundle_id": build_manifest_bundle_id(cfg.data.dataset_name),
    }

    bundle_dir = save_manifest_bundle(
        bundle_dir,
        run_manifest=run_manifest,
        stimulus_manifest=stimulus_manifest,
        friends_s7_manifest=friends_s7_manifest,
        ood_manifest=ood_manifest,
        metadata=metadata,
    )
    logger.info("Manifest bundle saved to %s", bundle_dir)
    logger.info(
        "Rows: run=%d stimulus=%d friends_s7=%d ood=%d",
        len(run_manifest),
        len(stimulus_manifest),
        len(friends_s7_manifest),
        len(ood_manifest),
    )


if __name__ == "__main__":
    main()
