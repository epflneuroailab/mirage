"""Generic challenge submission prediction helpers.

Supports the public Algonauts 2025 held-out benchmarks:

- Friends season 7 in-distribution benchmark
- OOD movie benchmark

Each benchmark is represented as a stimulus-only manifest. We segment each
stimulus into the same fixed 149 s windows used during training, run the model
window-by-window, concatenate the resulting 100-TR predictions per stimulus,
and trim to the organizer-provided ``target_sample_number``.
"""


import json
import logging
import shutil
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from brain_enc.checkpoints import load_model_state
from brain_enc.data.batch import _slice_feature_window, apply_pool_config
from brain_enc.data.feature_store import HDF5FeatureStore
from brain_enc.features.base import progress_iter
from brain_enc.modalities import MODALITIES

logger = logging.getLogger(__name__)

_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]
_FEATURE_HZ = 2.0


def _precision_forward_context(
    *,
    precision: str | None,
    device: str | torch.device,
):
    """Match training-time mixed precision for direct submission inference."""
    device_type = torch.device(device).type
    if device_type != "cuda" or precision is None:
        return nullcontext()

    normalized = str(precision).strip().lower()
    if "bf16" in normalized or "bfloat16" in normalized:
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    if (
        normalized.startswith("16")
        or "fp16" in normalized
        or "float16" in normalized
        or "16-mixed" in normalized
    ):
        return torch.autocast(device_type=device_type, dtype=torch.float16)
    return nullcontext()


def _predict_batch(
    model,
    batch_feats: dict[str, torch.Tensor],
    subj_batch: torch.Tensor,
    *,
    precision: str | None,
    device: str | torch.device,
    prediction_mode: str = "default",
) -> torch.Tensor:
    with torch.inference_mode(), _precision_forward_context(
        precision=precision,
        device=device,
    ):
        if prediction_mode != "default":
            return model(batch_feats, subj_batch, prediction_mode=prediction_mode)
        return model(batch_feats, subj_batch)


def _predict_subject_mean_batch(
    model,
    batch_feats: dict[str, torch.Tensor],
    subject_indices: list[int],
    *,
    precision: str | None,
    device: str | torch.device,
) -> torch.Tensor:
    """Average default subject-specific predictions over subject indices."""
    if not subject_indices:
        raise ValueError("subject_indices must be non-empty for subject_mean prediction.")
    preds: list[torch.Tensor] = []
    batch_size = next(iter(batch_feats.values())).shape[0]
    for subject_idx in subject_indices:
        subj_batch = torch.full(
            (batch_size,),
            int(subject_idx),
            dtype=torch.long,
            device=device,
        )
        preds.append(
            _predict_batch(
                model,
                batch_feats,
                subj_batch,
                precision=precision,
                device=device,
                prediction_mode="default",
            ).float()
        )
    return torch.stack(preds, dim=0).mean(dim=0)


def submission_zip_path(out_dir: str | Path, prediction_mode: str = "default") -> Path:
    """Return the zip filename for a submission prediction mode."""
    out_dir = Path(out_dir)
    if prediction_mode == "default":
        return out_dir / "submission.zip"
    safe_mode = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(prediction_mode).strip()
    )
    if not safe_mode:
        safe_mode = "mode"
    return out_dir / f"submission_{safe_mode}.zip"


@dataclass(frozen=True)
class SubmissionBenchmark:
    name: str
    output_dirname: str
    summary_filename: str
    sample_file_suffix: str
    manifest_builder_name: str

    def chunk_key(self, stimulus_id: str) -> str:
        return stimulus_id.split("/", 1)[1]

    def build_manifest(self, datapath: str | Path | None = None):
        from brain_enc.data import algonauts as algo

        builder = getattr(algo, self.manifest_builder_name)
        return builder(datapath)


def _load_submission_stimulus_manifest(
    *,
    cfg,
    bench: SubmissionBenchmark,
    datapath: Path,
):
    from brain_enc.data.manifest_io import maybe_load_manifest_bundle_for_config

    bundle = maybe_load_manifest_bundle_for_config(cfg)
    if bundle is not None and bench.name == "friends_s7":
        return bundle.friends_s7_manifest.copy(), bundle
    if bundle is not None and bench.name == "ood":
        return bundle.ood_manifest.copy(), bundle
    return bench.build_manifest(datapath), bundle


def _resolve_submission_feature_store(
    *,
    cfg,
    dataset_name: str,
    modality: str,
    cache_dir: str | Path | None,
):
    from brain_enc.config_schema import resolve_extractor_spec
    from brain_enc.paths import resolve_feature_store_for_config

    explicit_path = getattr(cfg.data, f"{modality}_h5_path")
    if explicit_path:
        resolved_spec = resolve_extractor_spec(
            getattr(cfg.data, modality),
            modality=modality,
        )
        return resolved_spec, Path(explicit_path)

    resolved_spec, path = resolve_feature_store_for_config(
        dataset_name,
        modality,
        getattr(cfg.data, modality),
        cache_dir=cache_dir,
    )
    return resolved_spec, path


_BENCHMARKS = {
    "friends_s7": SubmissionBenchmark(
        name="friends_s7",
        output_dirname="friends_s7",
        summary_filename="s7_prediction_summary.json",
        sample_file_suffix="friends-s7",
        manifest_builder_name="build_s7_manifest",
    ),
    "ood": SubmissionBenchmark(
        name="ood",
        output_dirname="ood",
        summary_filename="ood_prediction_summary.json",
        sample_file_suffix="ood",
        manifest_builder_name="build_ood_manifest",
    ),
}


def _resolve_run_name(run_dir: str | Path) -> str:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return run_dir.name

    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    if isinstance(raw, dict):
        return raw.get("run_name", run_dir.name)
    return run_dir.name


def get_submission_benchmark(name: str) -> SubmissionBenchmark:
    aliases = {
        "id_dist": "friends_s7",
        "in_distribution": "friends_s7",
        "in-distribution": "friends_s7",
        "friends": "friends_s7",
    }
    resolved = aliases.get(name, name)
    if resolved not in _BENCHMARKS:
        raise ValueError(
            f"Unknown submission benchmark {name!r}. "
            f"Available: {sorted(_BENCHMARKS)} plus aliases {sorted(aliases)}"
        )
    return _BENCHMARKS[resolved]


def require_submission_compatible_numpy() -> None:
    """Fail fast when saving with a NumPy major version the benchmark rejects."""
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        raise RuntimeError(
            "Challenge submissions must be saved with NumPy < 2.0, but the active "
            f"environment has NumPy {np.__version__}. Install 'numpy<2.0' and rerun "
            "the submission command."
        )


def resolve_sample_count_file(
    datapath: str | Path,
    subject: str,
    benchmark: str | SubmissionBenchmark,
) -> Path | None:
    """Resolve the organizer sample-count file for one benchmark + subject."""
    bench = (
        benchmark
        if isinstance(benchmark, SubmissionBenchmark)
        else get_submission_benchmark(benchmark)
    )
    datapath = Path(datapath)
    filename = f"{subject}_{bench.sample_file_suffix}_fmri_samples.npy"
    candidates = [
        datapath / "algonauts_2025.competitors" / "fmri" / subject / "target_sample_number" / filename,
        datapath / "fmri" / subject / "target_sample_number" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _write_submission_sidecar_artifacts(
    run_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, str | None]:
    """Write challenge-compatible sidecars next to ``submission.zip``."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = run_dir / "metrics.json"
    metrics_csv = out_dir / "metrics.csv"
    pearson_src = run_dir / "pearson_per_parcel.npy"
    pearson_dst = out_dir / "pearson.npy"

    written: dict[str, str | None] = {
        "metrics_csv": None,
        "pearson_npy": None,
    }

    if metrics_json.exists():
        import pandas as pd

        with open(metrics_json) as f:
            metrics = json.load(f)
        pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
        written["metrics_csv"] = str(metrics_csv)
    else:
        logger.warning("metrics.json not found in %s; metrics.csv sidecar not written.", run_dir)

    if pearson_src.exists():
        shutil.copy2(pearson_src, pearson_dst)
        written["pearson_npy"] = str(pearson_dst)
    else:
        logger.warning(
            "pearson_per_parcel.npy not found in %s; pearson.npy sidecar not written.",
            run_dir,
        )

    return written


def generate_submission_artifacts(
    *,
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    benchmark: str = "all",
    subjects: list[str] | None = None,
    batch_size: int = 8,
    datapath: str | Path | None = None,
    prediction_mode: str = "default",
) -> dict[str, dict]:
    """Generate one or more public benchmark submission bundles for a run."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    from brain_enc.eval.model_loading import default_checkpoint

    checkpoint = default_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No model.safetensors, best.ckpt, or last.ckpt in {run_dir}")

    from brain_enc.paths import submission_dir

    run_name = _resolve_run_name(run_dir)
    base_out_dir = Path(out_dir) if out_dir else submission_dir(run_name)
    if out_dir is None and prediction_mode != "default":
        base_out_dir = base_out_dir / prediction_mode
    benchmarks = (
        ["friends_s7", "ood"]
        if benchmark == "all"
        else [get_submission_benchmark(benchmark).name]
    )

    results: dict[str, dict] = {}
    benchmark_iter = progress_iter(
        benchmarks,
        desc="submission benchmarks",
        total=len(benchmarks),
        leave=True,
        unit="benchmark",
        position=0,
    )
    for benchmark_name in benchmark_iter:
        benchmark_iter.set_postfix_str(benchmark_name, refresh=False)
        if benchmark_name == "friends_s7":
            from brain_enc.eval.predict_friends_s7 import predict_friends_s7

            target_out_dir = (
                base_out_dir / "friends_s7"
                if benchmark == "all" or out_dir is None
                else base_out_dir
            )
            logger.info("Submission output dir (%s): %s", benchmark_name, target_out_dir)
            result = predict_friends_s7(
                run_dir=run_dir,
                out_dir=target_out_dir,
                datapath=datapath,
                subjects=subjects,
                batch_size=batch_size,
                prediction_mode=prediction_mode,
            )
        elif benchmark_name == "ood":
            from brain_enc.eval.predict_ood import predict_ood

            target_out_dir = (
                base_out_dir / "ood"
                if benchmark == "all" or out_dir is None
                else base_out_dir
            )
            logger.info("Submission output dir (%s): %s", benchmark_name, target_out_dir)
            result = predict_ood(
                run_dir=run_dir,
                out_dir=target_out_dir,
                datapath=datapath,
                subjects=subjects,
                batch_size=batch_size,
                prediction_mode=prediction_mode,
            )
        else:
            raise AssertionError(f"Unhandled benchmark {benchmark_name}")
        result["out_dir"] = target_out_dir
        results[benchmark_name] = result

    return results


def format_submission_results(results: dict[str, dict]) -> str:
    """Render a concise human-readable submission summary."""
    lines = ["", "=" * 60, "  Submission files generated", "=" * 60]
    for benchmark_name, result in results.items():
        lines.append(f"  Benchmark      : {benchmark_name}")
        if result.get("prediction_mode"):
            lines.append(f"  Prediction mode: {result['prediction_mode']}")
        submission_npy = result["submission_npy"]
        sub_dict = np.load(submission_npy, allow_pickle=True).item()
        for subject in sorted(sub_dict.keys()):
            chunks = sub_dict[subject]
            total_trs = sum(v.shape[0] for v in chunks.values())
            lines.append(f"  {subject}: {len(chunks)} chunks, {total_trs} total TRs")
        lines.append(f"  submission.npy : {submission_npy}")
        if result.get("submission_zip"):
            lines.append(f"  submission.zip : {result['submission_zip']}")
        if result.get("metrics_csv"):
            lines.append(f"  metrics.csv    : {result['metrics_csv']}")
        if result.get("pearson_npy"):
            lines.append(f"  pearson.npy    : {result['pearson_npy']}")
        lines.append(f"  Output dir     : {result['out_dir']}")
        lines.append("  " + "-" * 56)
    lines.append("=" * 60)
    return "\n".join(lines) + "\n"


def _load_feature_run(
    *,
    modality: str,
    stimulus_id: str,
    feature_stores: dict,
    pool_configs: dict[str, dict] | None,
    feature_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray | None]],
) -> tuple[np.ndarray, np.ndarray | None] | None:
    key = (modality, stimulus_id)
    if key in feature_cache:
        return feature_cache[key]

    store = feature_stores[modality]
    if not store.exists(stimulus_id):
        return None

    try:
        out = store.read(stimulus_id)
    except ValueError as exc:
        if (
            modality != "text"
            or "Per-layer feature group is empty" not in str(exc)
        ):
            raise
        # Legacy text caches for transcript-less OOD items were serialized under
        # the per-layer layout as an empty `features/` group. Submission should
        # treat them as empty text features and let the existing zero-fill path
        # handle them.
        time_axis = store.read_time_axis(stimulus_id)
        n_time = 0 if time_axis is None else int(len(time_axis))
        out = type(
            "SubmissionFeatureRun",
            (),
            {
                "features": np.zeros((0, n_time, 0), dtype=np.float32),
                "time_axis": time_axis,
            },
        )()
    pooled = apply_pool_config(
        out.features,
        None if pool_configs is None else pool_configs.get(modality),
    )
    feature_cache[key] = (pooled, out.time_axis)
    return feature_cache[key]


def _infer_feature_dims(
    *,
    stimulus_ids: list[str],
    feature_stores: dict,
    pool_configs: dict[str, dict] | None,
) -> dict[str, tuple[int, int]]:
    feature_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray | None]] = {}
    inferred: dict[str, tuple[int, int]] = {}

    for modality in feature_stores:
        for stimulus_id in stimulus_ids:
            loaded = _load_feature_run(
                modality=modality,
                stimulus_id=stimulus_id,
                feature_stores=feature_stores,
                pool_configs=pool_configs,
                feature_cache=feature_cache,
            )
            if loaded is None:
                continue
            features, _ = loaded
            if features.ndim >= 2 and features.shape[0] > 0:
                inferred[modality] = (int(features.shape[0]), int(features.shape[-1]))
                break
        if modality not in inferred:
            raise RuntimeError(
                f"Could not infer pooled feature dims for modality {modality!r} from "
                "the cached benchmark stimuli."
            )
    return inferred


def _complete_feature_dims(
    inferred: dict[str, tuple[int, int]],
) -> dict[str, tuple[int, int] | None]:
    """Return model feature dims with excluded modalities explicitly absent."""
    return {modality: inferred.get(modality) for modality in MODALITIES}


def prepare_segment_features(
    *,
    loaded_feature: tuple[np.ndarray, np.ndarray | None] | None,
    row: dict,
    expected_dims: tuple[int, int],
    allow_missing: bool,
) -> np.ndarray:
    """Convert a cached full-run feature tensor into one fixed inference window."""
    n_layers, n_dim = expected_dims
    n_frames = int(row["segment_n_feature_frames"])

    if loaded_feature is None:
        if not allow_missing:
            raise FileNotFoundError(
                f"Missing cached features for stimulus {row['stimulus_id']!r}."
            )
        return np.zeros((n_layers, n_frames, n_dim), dtype=np.float32)

    features, time_axis = loaded_feature
    if features.ndim < 3 or features.shape[0] == 0:
        if not allow_missing:
            raise RuntimeError(
                f"Cached features for stimulus {row['stimulus_id']!r} are empty."
            )
        return np.zeros((n_layers, n_frames, n_dim), dtype=np.float32)

    window = _slice_feature_window(
        features,
        time_axis,
        window_start_s=float(row["segment_start_s"]),
        window_duration_s=float(row["segment_duration_s"]),
        default_hz=_FEATURE_HZ,
    ).astype(np.float32, copy=False)
    if window.shape != (n_layers, n_frames, n_dim):
        raise RuntimeError(
            f"Unexpected window shape for {row['stimulus_id']!r}: got {window.shape}, "
            f"expected {(n_layers, n_frames, n_dim)}."
        )
    return window


def _load_segment_batch(
    rows: list[dict],
    *,
    feature_stores: dict,
    feature_dims: dict[str, tuple[int, int]],
    device: str,
    pool_configs: dict[str, dict] | None,
    feature_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray | None]],
) -> dict[str, torch.Tensor]:
    batch_feats: dict[str, torch.Tensor] = {}
    for modality in feature_stores:
        arrays: list[np.ndarray] = []
        missing_ids: list[str] = []
        for row in rows:
            loaded = _load_feature_run(
                modality=modality,
                stimulus_id=row["stimulus_id"],
                feature_stores=feature_stores,
                pool_configs=pool_configs,
                feature_cache=feature_cache,
            )
            allow_missing = modality == "text" and not (
                row.get("transcript_path", "") or row.get("transcript_relpath", "")
            )
            if loaded is None and not allow_missing:
                missing_ids.append(row["stimulus_id"])
                continue
            arrays.append(
                prepare_segment_features(
                    loaded_feature=loaded,
                    row=row,
                    expected_dims=feature_dims[modality],
                    allow_missing=allow_missing,
                )
            )
        if missing_ids:
            raise FileNotFoundError(
                f"Missing cached {modality} features for stimuli: {sorted(set(missing_ids))}. "
                "Run brain_enc.cli.extract_features first."
            )
        batch_feats[modality] = torch.from_numpy(np.stack(arrays, axis=0)).to(
            device,
            non_blocking=True,
        )
    return batch_feats


def _build_submission_feature_inputs(
    *,
    cfg,
    dataset_name: str,
    cache_dir,
    active_manifest_hash: str | None,
) -> tuple[dict, dict[str, dict]]:
    selected_modalities = list(cfg.data.modalities)
    pool_configs = {
        modality: {
            "layer_selection": getattr(cfg.input, modality).layer_selection,
            "layer_fractions": getattr(cfg.input, modality).layer_fractions,
            "layer_aggregation": getattr(cfg.input, modality).layer_aggregation,
        }
        for modality in selected_modalities
    }
    feature_stores = {}
    for modality in selected_modalities:
        resolved_spec, path = _resolve_submission_feature_store(
            cfg=cfg,
            dataset_name=dataset_name,
            modality=modality,
            cache_dir=cache_dir,
        )
        feature_stores[modality] = HDF5FeatureStore(
            path=path,
            extractor_id=resolved_spec.extractor_id,
            modality=modality,
            dataset_name=dataset_name,
            available_modalities=resolved_spec.available_modalities,
            expected_manifest_hash=active_manifest_hash,
        )
    return feature_stores, pool_configs


def predict_submission_benchmark(
    *,
    benchmark: str | SubmissionBenchmark,
    run_dir: str | Path,
    out_dir: str | Path,
    datapath: str | Path | None = None,
    subjects: list[str] | None = None,
    batch_size: int = 8,
    device: str | None = None,
    prediction_mode: str = "default",
) -> dict:
    """Generate challenge-format predictions for one public held-out benchmark."""
    bench = (
        benchmark
        if isinstance(benchmark, SubmissionBenchmark)
        else get_submission_benchmark(benchmark)
    )
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = subjects or _SUBJECTS
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    import yaml

    from brain_enc.config_schema import ExperimentConfig
    from brain_enc.data.algonauts import _SUBJECTS as ALL_SUBJECTS
    from brain_enc.data.algonauts import build_segment_manifest
    from brain_enc.models.builder import build_brain_model

    config_path = run_dir / "config.yaml"
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig(**raw)
    precision = cfg.training.precision

    cache_dir = cfg.data.hdf5_cache_dir

    if datapath is None:
        if cfg.data.datapath:
            datapath = Path(cfg.data.datapath)
        else:
            from brain_enc.env import get_datapath

            datapath = get_datapath()
    datapath = Path(datapath)

    stimulus_manifest, bundle = _load_submission_stimulus_manifest(
        cfg=cfg,
        bench=bench,
        datapath=datapath,
    )
    if stimulus_manifest.empty:
        raise RuntimeError(
            f"{bench.name} manifest is empty — no stimuli found under {datapath}."
        )
    segment_manifest = build_segment_manifest(stimulus_manifest)
    if segment_manifest.empty:
        raise RuntimeError(f"{bench.name} segment manifest is empty.")
    segment_manifest = segment_manifest.sort_values(
        ["stimulus_id", "segment_idx"]
    ).reset_index(drop=True)

    stimulus_ids = stimulus_manifest["stimulus_id"].tolist()
    chunk_keys = {sid: bench.chunk_key(sid) for sid in stimulus_ids}
    logger.info(
        "Predicting %s: %d stimuli expanded to %d windows for %d subjects",
        bench.name,
        len(stimulus_ids),
        len(segment_manifest),
        len(subjects),
    )

    dataset_name = cfg.data.dataset_name
    from brain_enc.data.manifest_io import manifest_bundle_hash

    active_manifest_hash = None if bundle is None else manifest_bundle_hash(bundle.metadata)
    feature_stores, pool_configs = _build_submission_feature_inputs(
        cfg=cfg,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        active_manifest_hash=active_manifest_hash,
    )

    feature_dims = _infer_feature_dims(
        stimulus_ids=stimulus_ids,
        feature_stores=feature_stores,
        pool_configs=pool_configs,
    )

    all_subjects = sorted(ALL_SUBJECTS)
    subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
    model = build_brain_model(
        cfg,
        feature_dims=_complete_feature_dims(feature_dims),
        n_parcels=1000,
        n_subjects=len(all_subjects),
    )

    from brain_enc.eval.model_loading import default_checkpoint

    checkpoint = default_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No model.safetensors, best.ckpt, or last.ckpt found in {run_dir}")
    load_model_state(model, checkpoint, map_location="cpu")
    model = model.to(device, non_blocking=True).eval()
    logger.info("Loaded checkpoint: %s", checkpoint)

    segment_rows = segment_manifest.to_dict("records")
    feature_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray | None]] = {}
    submission_dict: dict[str, dict[str, np.ndarray]] = {}

    valid_subjects = [subject for subject in subjects if subject in subject_to_idx]
    skipped_subjects = sorted(set(subjects) - set(valid_subjects))
    for subject in skipped_subjects:
        logger.warning("Unknown subject %s — skipping", subject)
    if not valid_subjects:
        raise ValueError(f"No known subjects requested. Known subjects: {all_subjects}")

    def _trim_subject_predictions(
        subject: str,
        subj_preds: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        samples_file = resolve_sample_count_file(datapath, subject, bench)
        if samples_file is None:
            logger.warning(
                "Could not find target_sample_number file for %s/%s; keeping full prediction lengths.",
                bench.name,
                subject,
            )
            return subj_preds

        target_sample_number: dict[str, int] = np.load(
            samples_file, allow_pickle=True
        ).item()
        expected_keys = set(target_sample_number)
        got_keys = set(subj_preds)
        if got_keys != expected_keys:
            missing = sorted(expected_keys - got_keys)
            extra = sorted(got_keys - expected_keys)
            raise ValueError(
                f"{bench.name}/{subject} submission keys mismatch. "
                f"Missing: {missing[:5]} Extra: {extra[:5]}"
            )
        trimmed = dict(subj_preds)
        for chunk_key, sample_number in target_sample_number.items():
            result = trimmed[chunk_key]
            if len(result) < sample_number:
                raise ValueError(
                    f"{len(result)} TR predictions for {subject}/{chunk_key} "
                    f"but target_sample_number={sample_number}"
                )
            trimmed[chunk_key] = result[:sample_number]
        logger.info(
            "Trimmed %s predictions to target_sample_number for %s",
            bench.name,
            subject,
        )
        return trimmed

    if prediction_mode == "subject_mean":
        subject_indices = [subject_to_idx[subject] for subject in valid_subjects]
        mean_chunks: dict[str, list[np.ndarray]] = {}
        n_batches = (len(segment_rows) + batch_size - 1) // batch_size
        batch_iter = progress_iter(
            range(0, len(segment_rows), batch_size),
            desc=f"{bench.name} batches subject_mean",
            total=n_batches,
            leave=True,
            unit="batch",
            position=0,
        )
        for i in batch_iter:
            batch_rows = segment_rows[i : i + batch_size]
            batch_iter.set_postfix_str(
                bench.chunk_key(batch_rows[0]["stimulus_id"]),
                refresh=False,
            )
            batch_feats = _load_segment_batch(
                batch_rows,
                feature_stores=feature_stores,
                feature_dims=feature_dims,
                device=device,
                pool_configs=pool_configs,
                feature_cache=feature_cache,
            )
            pred = _predict_subject_mean_batch(
                model,
                batch_feats,
                subject_indices,
                precision=precision,
                device=device,
            )  # (B, T=100, P)
            pred_np = pred.detach().cpu().numpy()
            for j, row in enumerate(batch_rows):
                chunk_key = chunk_keys[row["stimulus_id"]]
                chunk_pred = pred_np[j].astype(np.float32, copy=False)
                mean_chunks.setdefault(chunk_key, []).append(chunk_pred)

        mean_preds = {
            chunk_key: np.concatenate(parts, axis=0)
            for chunk_key, parts in mean_chunks.items()
        }
        for subject in valid_subjects:
            subject_preds = {
                chunk_key: value.astype(np.float32, copy=True)
                for chunk_key, value in mean_preds.items()
            }
            submission_dict[subject] = _trim_subject_predictions(subject, subject_preds)
        logger.info(
            "Accumulated %s subject-mean predictions: %d chunks reused for %d subjects",
            bench.name,
            len(mean_preds),
            len(submission_dict),
        )
    else:
        subject_iter = progress_iter(
            valid_subjects,
            desc=f"{bench.name} subjects",
            total=len(valid_subjects),
            leave=True,
            unit="subject",
            position=0,
        )
        for subject in subject_iter:
            subj_idx = subject_to_idx[subject]

            subject_id_tensor = torch.tensor([subj_idx], dtype=torch.long, device=device)
            subj_chunks: dict[str, list[np.ndarray]] = {}
            n_batches = (len(segment_rows) + batch_size - 1) // batch_size
            batch_iter = progress_iter(
                range(0, len(segment_rows), batch_size),
                desc=f"{bench.name} batches {subject}",
                total=n_batches,
                leave=False,
                unit="batch",
                position=1,
            )
            for i in batch_iter:
                batch_rows = segment_rows[i : i + batch_size]
                batch_iter.set_postfix_str(
                    bench.chunk_key(batch_rows[0]["stimulus_id"]),
                    refresh=False,
                )
                batch_feats = _load_segment_batch(
                    batch_rows,
                    feature_stores=feature_stores,
                    feature_dims=feature_dims,
                    device=device,
                    pool_configs=pool_configs,
                    feature_cache=feature_cache,
                )
                subj_batch = subject_id_tensor.expand(len(batch_rows))
                pred = _predict_batch(
                    model,
                    batch_feats,
                    subj_batch,
                    precision=precision,
                    device=device,
                    prediction_mode=prediction_mode,
                )  # (B, T=100, P)
                pred_np = pred.detach().cpu().numpy()
                for j, row in enumerate(batch_rows):
                    chunk_key = chunk_keys[row["stimulus_id"]]
                    chunk_pred = pred_np[j].astype(np.float32, copy=False)
                    subj_chunks.setdefault(chunk_key, []).append(chunk_pred)

            subj_preds = {
                chunk_key: np.concatenate(parts, axis=0)
                for chunk_key, parts in subj_chunks.items()
            }
            submission_dict[subject] = _trim_subject_predictions(subject, subj_preds)
            subject_iter.set_postfix_str(
                f"{subject}: {len(submission_dict[subject])} chunks",
                refresh=False,
            )
            logger.info(
                "Accumulated %s predictions for %s: %d chunks",
                bench.name,
                subject,
                len(submission_dict[subject]),
            )

    require_submission_compatible_numpy()
    submission_npy = out_dir / "submission.npy"
    np.save(submission_npy, submission_dict)
    logger.info("Saved submission dict → %s", submission_npy)

    submission_zip = submission_zip_path(out_dir, prediction_mode=prediction_mode)
    try:
        with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(submission_npy, arcname="submission.npy")
        logger.info("Zipped submission → %s", submission_zip)
    except Exception as exc:
        logger.warning("Could not create %s: %s", submission_zip.name, exc)
        submission_zip = None

    sidecars = _write_submission_sidecar_artifacts(run_dir, out_dir)
    summary = {
        "benchmark": bench.name,
        "run_dir": str(run_dir),
        "prediction_mode": prediction_mode,
        "n_stimuli": len(stimulus_ids),
        "n_windows": len(segment_rows),
        "subjects": list(submission_dict.keys()),
        "stimulus_ids": stimulus_ids,
        "submission_npy": str(submission_npy),
        "submission_zip": str(submission_zip) if submission_zip else None,
        "metrics_csv": sidecars["metrics_csv"],
        "pearson_npy": sidecars["pearson_npy"],
    }
    with open(out_dir / bench.summary_filename, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "benchmark": bench.name,
        "prediction_mode": prediction_mode,
        "submission_npy": submission_npy,
        "submission_zip": submission_zip,
        "metrics_csv": Path(sidecars["metrics_csv"]) if sidecars["metrics_csv"] else None,
        "pearson_npy": Path(sidecars["pearson_npy"]) if sidecars["pearson_npy"] else None,
        "subjects": list(submission_dict.keys()),
        "n_stimuli": len(stimulus_ids),
        "n_windows": len(segment_rows),
    }
