"""CLI: extract and cache modality features to HDF5 stores.

Run ``prepare_manifest`` first to build the canonical manifest bundle, then
invoke this CLI with the same config:

    python -m brain_enc.cli.prepare_manifest --config configs/experiments/mirage.yaml
    python -m brain_enc.cli.extract_features  --config configs/experiments/mirage.yaml

Useful flags: ``--modality``, ``--subjects``, ``--overwrite``,
``--stimulus-index``, ``--save-dtype``,
``--print-stimulus-count``, ``--list-stimuli``.

Stimulus caches store raw 2 Hz hidden-state sequences with no extraction-time
layer pooling. Fractional layer pooling is applied later at training time when
features are read back from HDF5.

Caches are written under
``$SCRATCHPATH/$DATASET_PATH/extracted_features/<extractor_id>/...``.
"""


import argparse
import logging
import sys
import typing as tp
from pathlib import Path
from typing import Sequence

from brain_enc.data.algonauts import build_stimulus_manifest as _build_stimulus_manifest
from brain_enc.cli.stimulus_index import (
    format_stimulus_index_listing as _format_stimulus_index_listing,
    format_stimulus_index_summary as _format_stimulus_index_summary,
    index_stimulus_manifest as _index_stimulus_manifest,
    select_stimulus_ids as _select_stimulus_ids,
)
from brain_enc.features.pipeline import (
    build_runtime_extractor_cfg as _build_runtime_extractor_cfg,
    extract_fmri,
    extract_joint_modalities,
    extract_modality,
)
from brain_enc.qwen_ids import resolve_qwen_extractor_id_for_causality as _resolve_qwen_extractor_id_for_causality

build_stimulus_manifest = _build_stimulus_manifest
format_stimulus_index_listing = _format_stimulus_index_listing
format_stimulus_index_summary = _format_stimulus_index_summary
index_stimulus_manifest = _index_stimulus_manifest
select_stimulus_ids = _select_stimulus_ids

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_MODALITIES = ["fmri", "text", "audio", "vision"]


_DTYPE_ALIASES = {
    "fp32": "float32",
    "f32": "float32",
    "float": "float32",
    "fp16": "float16",
    "f16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "bfloat": "bfloat16",
    "fp64": "float64",
    "f64": "float64",
    "double": "float64",
}
_DTYPE_CANONICAL = {"auto", "float32", "float16", "bfloat16", "float64"}
_SAVE_DTYPE_CANONICAL = {"source", "float32", "float16", "float64"}


def _parse_dtype_arg(value: str) -> str:
    normalized = str(value).strip().lower()
    normalized = _DTYPE_ALIASES.get(normalized, normalized)
    if normalized not in _DTYPE_CANONICAL:
        raise argparse.ArgumentTypeError(
            f"Unsupported dtype {value!r}. Expected one of "
            f"{sorted(_DTYPE_CANONICAL)} or aliases {sorted(_DTYPE_ALIASES)}."
        )
    return normalized


def _parse_save_dtype_arg(value: str) -> str:
    normalized = str(value).strip().lower()
    normalized = _DTYPE_ALIASES.get(normalized, normalized)
    if normalized not in _SAVE_DTYPE_CANONICAL:
        raise argparse.ArgumentTypeError(
            f"Unsupported save dtype {value!r}. Expected one of "
            f"{sorted(_SAVE_DTYPE_CANONICAL)} or aliases {sorted(_DTYPE_ALIASES)}."
        )
    return normalized


def _parse_bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value like True or False, got {value!r}"
    )


def _can_use_joint_windowed_qwen_path(
    modalities: Sequence[str],
    runtime_specs: dict[str, dict[str, tp.Any]],
) -> bool:
    allowed = {"text", "audio", "vision"}
    selected = set(modalities)
    if len(selected) < 2:
        return False
    if not selected.issubset(allowed):
        return False
    if set(runtime_specs) != selected:
        return False

    extractor_ids = {tp.cast(str, spec["extractor_id"]) for spec in runtime_specs.values()}
    if len(extractor_ids) != 1:
        return False
    extractor_id = next(iter(extractor_ids))
    if not extractor_id.endswith("_windowed"):
        return False

    expected_streams = {
        "text": "language_post_fusion",
        "audio": "audio_post_fusion",
        "vision": "vision_post_fusion",
    }
    shared_available: tuple[str, ...] | None = None
    for modality, spec in runtime_specs.items():
        if spec["stream_kind"] != expected_streams[modality]:
            return False
        extractor_cfg = tp.cast(dict[str, tp.Any], spec["extractor_cfg"])
        if bool(extractor_cfg.get("tower_only", False)):
            return False
        available_modalities = tp.cast(tuple[str, ...] | None, spec["available_modalities"])
        if available_modalities is None:
            return False
        if modality not in available_modalities:
            return False
        if shared_available is None:
            shared_available = available_modalities
        elif available_modalities != shared_available:
            return False

    return shared_available is not None


def _apply_cli_overrides(mod_cfg: tp.Any, **updates: tp.Any) -> tp.Any:
    """Return a validated config object with selected CLI overrides applied."""

    changed = {key: value for key, value in updates.items() if value is not None and hasattr(mod_cfg, key)}
    if not changed:
        return mod_cfg
    return type(mod_cfg).model_validate(
        {
            **mod_cfg.model_dump(),
            **changed,
        }
    )


def _resolve_system_prompt_arg(
    *,
    system_prompt: str | None,
    system_prompt_file: str | None,
) -> str | None:
    if system_prompt is not None:
        return system_prompt
    if system_prompt_file is None:
        return None
    return Path(system_prompt_file).read_text(encoding="utf-8")


def _resolve_requested_modalities(
    requested: Sequence[str] | None,
    cfg: tp.Any,
) -> list[str]:
    """Resolve extraction modalities from CLI input plus config defaults."""

    selected = list(requested) if requested else list(getattr(cfg.extract, "modalities", []) or [])
    if not selected:
        return list(ALL_MODALITIES)
    if "all" in selected:
        return list(ALL_MODALITIES)
    return selected


def _normalize_output_suffix_argv(argv: Sequence[str] | None) -> list[str]:
    """Rewrite ``--output-suffix VALUE`` into ``--output-suffix=VALUE``."""

    raw_args = list(sys.argv[1:] if argv is None else argv)
    normalized: list[str] = []
    i = 0
    while i < len(raw_args):
        token = raw_args[i]
        if token == "--output-suffix" and i + 1 < len(raw_args):
            normalized.append(f"--output-suffix={raw_args[i + 1]}")
            i += 2
            continue
        normalized.append(token)
        i += 1
    return normalized


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract and cache modality features to HDF5 stores."
    )
    parser.add_argument(
        "--config",
        required=True,
        nargs="+",
        help="One or more YAML configs merged in order. Later configs override earlier ones.",
    )
    parser.add_argument(
        "--modality",
        nargs="+",
        choices=ALL_MODALITIES + ["all"],
        default=None,
        help=(
            "Modalities to extract. When omitted, use extract.modalities from "
            "the config if present, otherwise default to all."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Restrict to specific subjects, e.g. sub-01 sub-02.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if the feature already exists in the store.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--stimulus-index",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Restrict extraction to specific zero-based stimulus indices from the "
            "stable unique-stimulus manifest. Intended for external job sharding."
        ),
    )
    parser.add_argument(
        "--print-stimulus-count",
        action="store_true",
        help="Print the stable unique-stimulus count and exit.",
    )
    parser.add_argument(
        "--list-stimuli",
        action="store_true",
        help="Print the full stable stimulus index map and exit.",
    )
    parser.add_argument(
        "--context-modalities",
        "--available-modalities",
        dest="context_modalities",
        nargs="+",
        choices=["text", "audio", "vision"],
        default=None,
        help=(
            "Override the configured conditioning subset for stimulus extractors. "
            "For audio and vision targets, the selected target modality must be "
            "included. Text-target Qwen runs may omit text to extract "
            "language-module features from media-only context."
        ),
    )
    parser.add_argument(
        "--stream-kind",
        choices=[
            "auto",
            "language_post_fusion",
            "language_module_from_media",
            "audio_post_fusion",
            "audio_tower",
            "vision_post_fusion",
            "vision_tower",
        ],
        default=None,
        help=(
            "Override the resolved exported-stream identifier used in cache "
            "paths and metadata. In most cases leave this unset and rely on "
            "the mode inferred from the extractor plus context modalities."
        ),
    )
    parser.add_argument(
        "--multimodal-causal",
        "--qwen-causal",
        dest="multimodal_causal",
        type=_parse_bool_arg,
        default=None,
        metavar="{True,False}",
        help=(
            "For Qwen extractors, explicitly switch between the causal cache "
            "identities (`True`) and the preserved non-causal baseline "
            "(`False`). When unset, preserve the extractor IDs declared in the "
            "config."
        ),
    )
    parser.add_argument(
        "--modality-tower-only",
        "--qwen-tower-only",
        dest="modality_tower_only",
        type=_parse_bool_arg,
        default=None,
        metavar="{True,False}",
        help=(
            "For Qwen audio/vision extraction, optionally bypass the fused "
            "language stack and cache only modality-tower features with the "
            "same temporal windowing semantics. The cache filename changes "
            "when this is enabled. `--qwen-tower-only` remains as a backward-"
            "compatible alias."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=_parse_dtype_arg,
        default=None,
        help=(
            "Override the extractor dtype for text/audio/vision modalities. "
            "Accepts canonical names (auto, float32, float16, bfloat16, "
            "float64) or short aliases (fp32, fp16, bf16, fp64, half, double). "
            "Equivalent to setting data.<modality>.dtype on every stimulus modality."
        ),
    )
    parser.add_argument(
        "--save-dtype",
        type=_parse_save_dtype_arg,
        default="float16",
        help=(
            "Cast extracted features just before saving to HDF5. This does not "
            "change model/extractor compute precision. Applies to stimulus "
            "features and fMRI. Use 'source' to preserve the extractor's "
            "original dtype."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help=(
            "Optional string appended to the output feature-store filename "
            "immediately before `.h5`, e.g. `text-ablation.h5` or "
            "`ctx-audio-text--trial.h5`."
        ),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["manual", "chat_template"],
        default=None,
        help=(
            "For Qwen extractors, optionally render the text prompt via the "
            "Hugging Face chat template instead of the legacy manual prompt."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help=(
            "Optional system prompt used only with --prompt-mode chat_template. "
            "Prompt identity is encoded in the cache filename and metadata."
        ),
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help=(
            "Read the system prompt from a UTF-8 text file. Ignored when "
            "--system-prompt is set explicitly."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Key=value config overrides.",
    )
    args = parser.parse_args(_normalize_output_suffix_argv(argv))

    # Load config
    from brain_enc.cli._paths import resolve_config_paths
    from brain_enc.config_schema import load_config, resolve_extractor_spec

    config_paths = resolve_config_paths(args.config)
    logger.info("Config: %s", ", ".join(str(path) for path in config_paths))
    cfg = load_config(config_paths, overrides=args.overrides)
    cfg.resolve_paths()
    modalities = _resolve_requested_modalities(args.modality, cfg)
    system_prompt = _resolve_system_prompt_arg(
        system_prompt=args.system_prompt,
        system_prompt_file=args.system_prompt_file,
    )
    logger.info("Requested modalities: %s", ", ".join(modalities))

    # Resolve canonical manifest bundle
    from brain_enc.data.manifest_io import (
        build_extraction_stimulus_manifest,
        manifest_bundle_hash,
        maybe_load_manifest_bundle_for_config,
        resolve_bundle_manifest_paths,
    )
    from brain_enc.env import get_datapath
    from brain_enc.paths import feature_store_path

    bundle = maybe_load_manifest_bundle_for_config(cfg)
    if bundle is None:
        raise FileNotFoundError(
            "Feature extraction now requires a canonical manifest bundle. "
            "Run `python -m brain_enc.cli.prepare_manifest --config ...` first, "
            "or set data.manifest_dir to an existing bundle."
        )

    datapath = Path(cfg.data.datapath) if cfg.data.datapath else get_datapath()
    logger.info("Dataset root: %s", datapath)
    logger.info("Using manifest bundle: %s", bundle.bundle_dir)
    manifest = resolve_bundle_manifest_paths(bundle.run_manifest, datapath=datapath)
    stimulus_manifest = resolve_bundle_manifest_paths(
        build_extraction_stimulus_manifest(bundle),
        datapath=datapath,
    )
    indexed_stimuli = _index_stimulus_manifest(stimulus_manifest)
    active_manifest_hash = manifest_bundle_hash(bundle.metadata)

    if args.print_stimulus_count:
        print(_format_stimulus_index_summary(indexed_stimuli))
        return

    if args.list_stimuli:
        print(_format_stimulus_index_listing(indexed_stimuli))
        return

    selected_stimulus_ids = _select_stimulus_ids(stimulus_manifest, args.stimulus_index)

    if selected_stimulus_ids is not None:
        manifest = manifest[manifest["stimulus_id"].isin(selected_stimulus_ids)]
        stimulus_manifest = stimulus_manifest[
            stimulus_manifest["stimulus_id"].isin(selected_stimulus_ids)
        ]
        indexed_stimuli = _index_stimulus_manifest(stimulus_manifest)
        logger.info(
            "Filtered manifest to %d rows across %d selected stimuli",
            len(manifest),
            len(selected_stimulus_ids),
        )

    # Restrict to requested subjects
    if args.subjects:
        manifest = manifest[manifest["subject"].isin(args.subjects)]
        logger.info("Filtered to subjects: %s  (%d rows)", args.subjects, len(manifest))

    dataset_name = cfg.data.dataset_name
    cache_dir = cfg.data.hdf5_cache_dir  # None -> default dataset-root extracted_features path

    # --- fMRI ---
    if "fmri" in modalities:
        logger.info("=== Extracting fMRI ===")
        store_path = feature_store_path(
            dataset_name,
            "fmri",
            dataset_name,
            cache_dir=cache_dir,
            output_suffix=args.output_suffix,
        )
        extract_fmri(
            manifest,
            store_path=store_path,
            overwrite=args.overwrite,
            n_workers=args.n_workers,
            dataset_name=dataset_name,
            save_dtype=args.save_dtype,
            manifest_hash=active_manifest_hash,
        )

    # --- Stimulus modalities ---
    modality_cfg_map = {
        "text":   (cfg.data.text.name,   cfg.data.text),
        "audio":  (cfg.data.audio.name,  cfg.data.audio),
        "vision": (cfg.data.vision.name, cfg.data.vision),
    }
    resolved_extractor_ids: dict[str, str] = {}
    resolved_available_modalities: dict[str, list[str] | None] = {}
    resolved_stream_kinds: dict[str, str | None] = {}
    resolved_cache_variants: dict[str, str | None] = {}
    resolved_prompt_ids: dict[str, str | None] = {}
    runtime_specs: dict[str, dict[str, tp.Any]] = {}
    for mod in ["text", "audio", "vision"]:
        if mod not in modalities:
            continue
        configured_extractor_id, mod_cfg = modality_cfg_map[mod]
        selected_available_modalities = (
            args.context_modalities
            if args.context_modalities is not None
            else getattr(mod_cfg, "available_modalities", None)
        )
        mod_cfg = _apply_cli_overrides(
            mod_cfg,
            tower_only=args.modality_tower_only,
            dtype=args.dtype,
            prompt_mode=args.prompt_mode,
            system_prompt=system_prompt,
        )
        resolved_spec = resolve_extractor_spec(
            mod_cfg,
            modality=tp.cast(tp.Any, mod),
            multimodal_causal=args.multimodal_causal,
            available_modalities=selected_available_modalities,
            stream_kind=args.stream_kind,
        )
        if resolved_spec.extractor_id != configured_extractor_id:
            logger.warning(
                "%s: switching extractor %s -> %s via explicit --multimodal-causal=%s",
                mod,
                configured_extractor_id,
                resolved_spec.extractor_id,
                args.multimodal_causal,
            )
        extractor_id = resolved_spec.extractor_id
        resolved_mod_cfg = resolved_spec.config
        cache_available_modalities = resolved_spec.available_modalities
        cache_stream_kind = resolved_spec.stream_kind
        resolved_extractor_ids[mod] = extractor_id
        resolved_available_modalities[mod] = (
            None if cache_available_modalities is None else list(cache_available_modalities)
        )
        resolved_stream_kinds[mod] = cache_stream_kind
        resolved_cache_variants[mod] = resolved_spec.cache_variant
        resolved_prompt_ids[mod] = resolved_spec.prompt_id
        store_path = feature_store_path(
            dataset_name,
            mod,
            extractor_id,
            available_modalities=cache_available_modalities,
            stream_kind=cache_stream_kind,
            cache_variant=resolved_cache_variants[mod],
            prompt_id=resolved_prompt_ids[mod],
            cache_dir=cache_dir,
            output_suffix=args.output_suffix,
        )
        extractor_cfg = _build_runtime_extractor_cfg(
            extractor_id,
            resolved_mod_cfg,
            available_modalities=cache_available_modalities,
        )
        runtime_specs[mod] = {
            "extractor_id": extractor_id,
            "extractor_cfg": extractor_cfg,
            "store_path": store_path,
            "available_modalities": cache_available_modalities,
            "stream_kind": cache_stream_kind,
            "layer_fractions": getattr(getattr(cfg.input, mod), "layer_fractions", None),
            "layer_aggregation": getattr(getattr(cfg.input, mod), "layer_aggregation", None),
        }

    if _can_use_joint_windowed_qwen_path(modalities, runtime_specs):
        logger.info(
            "=== Extracting %s via joint Qwen windowed shared-forward path ===",
            "/".join(mod for mod in ["text", "audio", "vision"] if mod in modalities),
        )
        extract_joint_modalities(
            indexed_stimuli,
            modality_specs=runtime_specs,
            overwrite=args.overwrite,
            n_workers=args.n_workers,
            dataset_name=dataset_name,
            save_dtype=args.save_dtype,
            manifest_hash=active_manifest_hash,
        )
    else:
        for mod in ["text", "audio", "vision"]:
            if mod not in modalities:
                continue
            logger.info("=== Extracting %s ===", mod)
            spec = runtime_specs[mod]
            extract_modality(
                indexed_stimuli,
                modality=mod,
                extractor_id=spec["extractor_id"],
                extractor_cfg=spec["extractor_cfg"],
                available_modalities=spec["available_modalities"],
                stream_kind=spec["stream_kind"],
                store_path=spec["store_path"],
                overwrite=args.overwrite,
                n_workers=args.n_workers,
                dataset_name=dataset_name,
                layer_fractions=spec["layer_fractions"],
                layer_aggregation=spec["layer_aggregation"],
                save_dtype=args.save_dtype,
                manifest_hash=active_manifest_hash,
            )

    logger.info("Feature extraction complete.")

    # Write a feature-cache inspection summary (phase-1 deliverable)
    from brain_enc.eval.benchmark import inspect_feature_cache

    extractor_map: dict[str, str | dict[str, tp.Any]] = {"fmri": dataset_name}
    for mod in ("text", "audio", "vision"):
        mod_cfg = getattr(cfg.data, mod)
        cache_available_modalities = getattr(mod_cfg, "cache_available_modalities", None)
        extractor_map[mod] = {
            "extractor_id": resolved_extractor_ids.get(mod, mod_cfg.name),
            "available_modalities": resolved_available_modalities.get(
                mod,
                None if cache_available_modalities is None else list(cache_available_modalities),
            ),
            "stream_kind": resolved_stream_kinds.get(mod, getattr(mod_cfg, "cache_stream_kind", None)),
            "cache_variant": resolved_cache_variants.get(mod, getattr(mod_cfg, "cache_variant", None)),
            "prompt_id": resolved_prompt_ids.get(mod, getattr(mod_cfg, "cache_prompt_id", None)),
        }
    # Only report modalities that were (or could have been) extracted this run
    run_map = {m: extractor_map[m] for m in modalities if m in extractor_map}
    cache_summary_path = (
        Path(cfg.run_dir) / "feature_cache_summary.json"
        if cfg.run_dir
        else None
    )
    inspect_feature_cache(
        dataset_name=dataset_name,
        extractor_map=run_map,
        cache_dir=cache_dir,
        out_path=cache_summary_path,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
