"""TSV-backed manifest bundle IO and validation helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import typing as tp

import numpy as np
import pandas as pd

from brain_enc.data.algonauts import annotate_manifest_durations
from brain_enc.paths import manifest_root


MANIFEST_SCHEMA_VERSION = "v1"
MANIFEST_BUNDLE_FILENAME = "manifest_bundle.json"
RUN_MANIFEST_FILENAME = "run_manifest.tsv"
STIMULUS_MANIFEST_FILENAME = "stimulus_manifest.tsv"
FRIENDS_S7_MANIFEST_FILENAME = "friends_s7_manifest.tsv"
OOD_MANIFEST_FILENAME = "ood_manifest.tsv"

RUN_MANIFEST_COLUMNS = (
    "dataset_name",
    "stimulus_namespace",
    "subject",
    "subject_idx",
    "task",
    "movie",
    "stimulus_id",
    "fmri_item_id",
    "chunk",
    "fmri_relpath",
    "fmri_h5_key",
    "video_relpath",
    "transcript_relpath",
    "audio_relpath",
    "n_parcels",
    "transcript_duration_s",
    "video_duration_s",
    "audio_duration_s",
    "fmri_duration_s",
)
STIMULUS_MANIFEST_COLUMNS = (
    "dataset_name",
    "stimulus_namespace",
    "task",
    "movie",
    "stimulus_id",
    "chunk",
    "video_relpath",
    "transcript_relpath",
    "audio_relpath",
    "transcript_duration_s",
    "video_duration_s",
    "audio_duration_s",
)
BENCHMARK_MANIFEST_COLUMNS = (
    "dataset_name",
    "stimulus_namespace",
    "task",
    "movie",
    "stimulus_id",
    "chunk",
    "video_relpath",
    "transcript_relpath",
    "audio_relpath",
    "transcript_duration_s",
    "video_duration_s",
    "audio_duration_s",
)
_BUNDLE_METADATA_REQUIRED_KEYS = (
    "manifest_schema_version",
    "dataset_name",
    "stimulus_namespace",
    "bundle_id",
    "created_at_utc",
    "row_counts",
    "hashes",
)


@dataclasses.dataclass
class ManifestBundle:
    bundle_dir: Path
    metadata: dict[str, tp.Any]
    run_manifest: pd.DataFrame
    stimulus_manifest: pd.DataFrame
    friends_s7_manifest: pd.DataFrame
    ood_manifest: pd.DataFrame


@dataclasses.dataclass
class RunManifestBundle:
    bundle_dir: Path
    metadata: dict[str, tp.Any]
    run_manifest: pd.DataFrame


def _canonicalize_manifest_df(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = sorted(df.columns.tolist())
    return df.loc[:, sort_columns].sort_values(sort_columns).reset_index(drop=True)


def hash_manifest_df(df: pd.DataFrame) -> str:
    canonical = _canonicalize_manifest_df(df)
    payload = canonical.to_csv(sep="\t", index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest_hashes(
    *,
    run_manifest: pd.DataFrame,
    stimulus_manifest: pd.DataFrame,
    friends_s7_manifest: pd.DataFrame,
    ood_manifest: pd.DataFrame,
) -> dict[str, str]:
    hashes = {
        "run_manifest_sha256": hash_manifest_df(run_manifest),
        "stimulus_manifest_sha256": hash_manifest_df(stimulus_manifest),
        "friends_s7_manifest_sha256": hash_manifest_df(friends_s7_manifest),
        "ood_manifest_sha256": hash_manifest_df(ood_manifest),
    }
    hashes["bundle_sha256"] = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return hashes


def enrich_manifest_metadata_with_hashes(
    metadata: dict[str, tp.Any],
    *,
    run_manifest: pd.DataFrame,
    stimulus_manifest: pd.DataFrame,
    friends_s7_manifest: pd.DataFrame,
    ood_manifest: pd.DataFrame,
) -> dict[str, tp.Any]:
    payload = dict(metadata)
    payload["hashes"] = build_manifest_hashes(
        run_manifest=run_manifest,
        stimulus_manifest=stimulus_manifest,
        friends_s7_manifest=friends_s7_manifest,
        ood_manifest=ood_manifest,
    )
    return payload


def manifest_bundle_hash(metadata: dict[str, tp.Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    hashes = metadata.get("hashes")
    if not isinstance(hashes, dict):
        return None
    value = hashes.get("bundle_sha256")
    return value if isinstance(value, str) and value else None


def build_manifest_bundle_id(
    dataset_name: str,
    *,
    schema_version: str = MANIFEST_SCHEMA_VERSION,
) -> str:
    return f"dataset-{dataset_name}__schema-{schema_version}"


def default_manifest_dir(cfg) -> Path:
    return (
        manifest_root(datapath=cfg.data.datapath, cache_dir=cfg.data.hdf5_cache_dir)
        / cfg.data.dataset_name
        / build_manifest_bundle_id(cfg.data.dataset_name)
    )


def maybe_load_manifest_bundle_for_config(cfg) -> ManifestBundle | None:
    if cfg.data.manifest_dir:
        return load_manifest_bundle(Path(cfg.data.manifest_dir))
    bundle_dir = default_manifest_dir(cfg)
    if bundle_dir.exists():
        return load_manifest_bundle(bundle_dir)
    return None


def _bundle_paths(bundle_dir: Path) -> dict[str, Path]:
    return {
        "metadata": bundle_dir / MANIFEST_BUNDLE_FILENAME,
        "run_manifest": bundle_dir / RUN_MANIFEST_FILENAME,
        "stimulus_manifest": bundle_dir / STIMULUS_MANIFEST_FILENAME,
        "friends_s7_manifest": bundle_dir / FRIENDS_S7_MANIFEST_FILENAME,
        "ood_manifest": bundle_dir / OOD_MANIFEST_FILENAME,
    }


def _empty_audio_path(df: pd.DataFrame) -> pd.Series:
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def _relativize_path(value: tp.Any, *, datapath: Path) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        return raw
    try:
        return str(path.resolve().relative_to(datapath.resolve()))
    except ValueError as exc:
        raise ValueError(
            f"Canonical manifest paths must stay under datapath {datapath}, got {path}"
        ) from exc


def _canonicalize_relpaths(df: pd.DataFrame, *, datapath: str | Path | None) -> pd.DataFrame:
    if datapath is None:
        raise ValueError("datapath is required to canonicalize manifest paths.")
    root = Path(datapath)
    out = df.copy()
    column_mapping = (
        ("fmri_h5_path", "fmri_relpath"),
        ("video_path", "video_relpath"),
        ("transcript_path", "transcript_relpath"),
        ("audio_path", "audio_relpath"),
    )
    for source_column, target_column in column_mapping:
        source = source_column if source_column in out.columns else target_column
        if source not in out.columns:
            out[target_column] = ""
            continue
        out[target_column] = out[source].map(
            lambda value: _relativize_path(value, datapath=root)
        ).astype("string")
    drop_columns = [column for column, _ in column_mapping if column in out.columns]
    return out.drop(columns=drop_columns, errors="ignore")


def _with_common_manifest_fields(df: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    out["dataset_name"] = dataset_name
    out["stimulus_namespace"] = dataset_name
    if "audio_path" not in out.columns and "audio_relpath" not in out.columns:
        out["audio_relpath"] = _empty_audio_path(out)
    return out


def _ensure_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            continue
        if column.endswith("_duration_s"):
            out[column] = 0.0
        elif column in {"subject_idx", "n_parcels"}:
            out[column] = 0
        else:
            out[column] = ""
    return out


def prepare_canonical_run_manifest(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    datapath: str | Path | None,
) -> pd.DataFrame:
    out = _with_common_manifest_fields(df, dataset_name=dataset_name)
    if not {
        "transcript_duration_s",
        "video_duration_s",
        "audio_duration_s",
        "fmri_duration_s",
    }.issubset(out.columns):
        out = annotate_manifest_durations(out)
    out = _canonicalize_relpaths(out, datapath=datapath)
    out = _ensure_columns(out, RUN_MANIFEST_COLUMNS)
    return out.loc[:, list(RUN_MANIFEST_COLUMNS)].copy()


def prepare_canonical_stimulus_manifest(
    run_manifest: pd.DataFrame,
    *,
    dataset_name: str,
    datapath: str | Path | None,
) -> pd.DataFrame:
    out = _with_common_manifest_fields(run_manifest, dataset_name=dataset_name)
    out = out.drop_duplicates(subset=["stimulus_namespace", "stimulus_id"]).copy()
    if not {"transcript_duration_s", "video_duration_s", "audio_duration_s"}.issubset(out.columns):
        out = annotate_manifest_durations(out)
    out = _canonicalize_relpaths(out, datapath=datapath)
    out = _ensure_columns(out, STIMULUS_MANIFEST_COLUMNS)
    return out.loc[:, list(STIMULUS_MANIFEST_COLUMNS)].copy()


def prepare_canonical_benchmark_manifest(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    datapath: str | Path | None,
) -> pd.DataFrame:
    out = _with_common_manifest_fields(df, dataset_name=dataset_name)
    if not {"transcript_duration_s", "video_duration_s", "audio_duration_s"}.issubset(out.columns):
        out = annotate_manifest_durations(out)
    out = out.drop_duplicates(subset=["stimulus_namespace", "stimulus_id"]).copy()
    out = _canonicalize_relpaths(out, datapath=datapath)
    out = _ensure_columns(out, BENCHMARK_MANIFEST_COLUMNS)
    return out.loc[:, list(BENCHMARK_MANIFEST_COLUMNS)].copy()


def _tsv_kwargs() -> dict[str, tp.Any]:
    return {"sep": "\t", "index": False}


def save_manifest_bundle(
    bundle_dir: str | Path,
    *,
    run_manifest: pd.DataFrame,
    stimulus_manifest: pd.DataFrame,
    friends_s7_manifest: pd.DataFrame,
    ood_manifest: pd.DataFrame,
    metadata: dict[str, tp.Any],
) -> Path:
    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)
    paths = _bundle_paths(bundle_path)

    run_manifest.to_csv(paths["run_manifest"], **_tsv_kwargs())
    stimulus_manifest.to_csv(paths["stimulus_manifest"], **_tsv_kwargs())
    friends_s7_manifest.to_csv(paths["friends_s7_manifest"], **_tsv_kwargs())
    ood_manifest.to_csv(paths["ood_manifest"], **_tsv_kwargs())

    payload = dict(metadata)
    payload.setdefault("manifest_schema_version", MANIFEST_SCHEMA_VERSION)
    payload.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    payload["row_counts"] = {
        "run_manifest": int(len(run_manifest)),
        "stimulus_manifest": int(len(stimulus_manifest)),
        "friends_s7_manifest": int(len(friends_s7_manifest)),
        "ood_manifest": int(len(ood_manifest)),
    }
    payload = enrich_manifest_metadata_with_hashes(
        payload,
        run_manifest=run_manifest,
        stimulus_manifest=stimulus_manifest,
        friends_s7_manifest=friends_s7_manifest,
        ood_manifest=ood_manifest,
    )
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return bundle_path


def _dtype_map(kind: str) -> dict[str, str]:
    string = "string"
    if kind == "run":
        return {
            "dataset_name": string,
            "stimulus_namespace": string,
            "subject": string,
            "task": string,
            "movie": string,
            "stimulus_id": string,
            "fmri_item_id": string,
            "chunk": string,
            "fmri_relpath": string,
            "fmri_h5_key": string,
            "video_relpath": string,
            "transcript_relpath": string,
            "audio_relpath": string,
        }
    if kind == "stimulus":
        return {
            "dataset_name": string,
            "stimulus_namespace": string,
            "task": string,
            "movie": string,
            "stimulus_id": string,
            "chunk": string,
            "video_relpath": string,
            "transcript_relpath": string,
            "audio_relpath": string,
        }
    if kind in {"friends_s7", "ood"}:
        return _dtype_map("stimulus")
    raise ValueError(f"Unsupported manifest kind: {kind}")


def _numeric_dtype_map(kind: str) -> dict[str, str]:
    if kind == "run":
        return {
            "subject_idx": "int64",
            "n_parcels": "int64",
            "transcript_duration_s": "float64",
            "video_duration_s": "float64",
            "audio_duration_s": "float64",
            "fmri_duration_s": "float64",
        }
    if kind in {"stimulus", "friends_s7", "ood"}:
        return {
            "transcript_duration_s": "float64",
            "video_duration_s": "float64",
            "audio_duration_s": "float64",
        }
    raise ValueError(f"Unsupported manifest kind: {kind}")


def _missing_required_keys(mapping: dict[str, tp.Any], required: tuple[str, ...]) -> list[str]:
    return [key for key in required if key not in mapping]


def _validate_manifest_schema_version(metadata: dict[str, tp.Any]) -> None:
    schema_version = metadata.get("manifest_schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported manifest_schema_version {schema_version!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION!r}."
        )


def validate_manifest_metadata(metadata: tp.Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("manifest metadata must be a mapping.")
    missing = _missing_required_keys(metadata, _BUNDLE_METADATA_REQUIRED_KEYS)
    if missing:
        raise ValueError(f"manifest metadata is missing required keys: {missing}")
    _validate_manifest_schema_version(metadata)

    row_counts = metadata.get("row_counts")
    if not isinstance(row_counts, dict):
        raise ValueError("row_counts must be a mapping.")
    missing_row_counts = _missing_required_keys(
        row_counts,
        ("run_manifest", "stimulus_manifest", "friends_s7_manifest", "ood_manifest"),
    )
    if missing_row_counts:
        raise ValueError(f"row_counts is missing required keys: {missing_row_counts}")
    for key in ("run_manifest", "stimulus_manifest", "friends_s7_manifest", "ood_manifest"):
        value = row_counts.get(key)
        if not isinstance(value, int):
            raise ValueError(f"row_counts[{key!r}] must be an integer.")
    hashes = metadata.get("hashes")
    required_hashes = (
        "run_manifest_sha256",
        "stimulus_manifest_sha256",
        "friends_s7_manifest_sha256",
        "ood_manifest_sha256",
        "bundle_sha256",
    )
    if not isinstance(hashes, dict):
        raise ValueError("hashes must be a mapping.")
    missing_hashes = _missing_required_keys(hashes, required_hashes)
    if missing_hashes:
        raise ValueError(f"hashes is missing required keys: {missing_hashes}")
    for key in required_hashes:
        value = hashes.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"hashes[{key!r}] must be a non-empty string.")


def _validate_required_numeric_columns(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    manifest_name: str,
) -> None:
    for column in columns:
        series = df[column]
        if series.isna().any():
            raise ValueError(f"{manifest_name}.{column} contains missing values")
        if not np.isfinite(series.to_numpy(dtype="float64", na_value=np.nan)).all():
            raise ValueError(f"{manifest_name}.{column} contains non-finite values")


def validate_run_manifest_df(
    manifest: pd.DataFrame,
    *,
    metadata: dict[str, tp.Any] | None = None,
) -> None:
    missing = [column for column in RUN_MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"run_manifest is missing required columns: {missing}")
    if metadata is not None:
        validate_manifest_metadata(metadata)
        expected = metadata["row_counts"]["run_manifest"]
        actual = int(len(manifest))
        if expected != actual:
            raise ValueError(f"run_manifest row count mismatch: expected {expected}, found {actual}")
        expected_hash = metadata["hashes"]["run_manifest_sha256"]
        actual_hash = hash_manifest_df(manifest)
        if expected_hash != actual_hash:
            raise ValueError("run_manifest hash mismatch")
    _validate_required_numeric_columns(
        manifest,
        columns=(
            "subject_idx",
            "n_parcels",
            "transcript_duration_s",
            "video_duration_s",
            "audio_duration_s",
            "fmri_duration_s",
        ),
        manifest_name="run_manifest",
    )
    if manifest.duplicated(subset=["subject", "stimulus_namespace", "stimulus_id"]).any():
        raise ValueError("run_manifest contains duplicate subject/stimulus rows")


def load_tsv_with_schema(
    path: str | Path,
    kind: tp.Literal["run", "stimulus", "friends_s7", "ood"],
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=_dtype_map(kind),
        keep_default_na=False,
    )
    for column, dtype in _numeric_dtype_map(kind).items():
        parsed = pd.to_numeric(df[column], errors="coerce")
        if dtype == "int64" and not parsed.isna().any():
            df[column] = parsed.astype("int64")
        else:
            df[column] = parsed.astype("float64")
    return df


def validate_manifest_bundle(bundle: ManifestBundle) -> None:
    validate_manifest_metadata(bundle.metadata)

    expected_columns = {
        "run_manifest": RUN_MANIFEST_COLUMNS,
        "stimulus_manifest": STIMULUS_MANIFEST_COLUMNS,
        "friends_s7_manifest": BENCHMARK_MANIFEST_COLUMNS,
        "ood_manifest": BENCHMARK_MANIFEST_COLUMNS,
    }
    for name, columns in expected_columns.items():
        df = getattr(bundle, name)
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    row_counts = bundle.metadata.get("row_counts", {})
    for name in expected_columns:
        expected = row_counts.get(name)
        actual = int(len(getattr(bundle, name)))
        if int(expected) != actual:
            raise ValueError(f"{name} row count mismatch: expected {expected}, found {actual}")

    validate_run_manifest_df(bundle.run_manifest)
    _validate_required_numeric_columns(
        bundle.stimulus_manifest,
        columns=("transcript_duration_s", "video_duration_s", "audio_duration_s"),
        manifest_name="stimulus_manifest",
    )
    _validate_required_numeric_columns(
        bundle.friends_s7_manifest,
        columns=("transcript_duration_s", "video_duration_s", "audio_duration_s"),
        manifest_name="friends_s7_manifest",
    )
    _validate_required_numeric_columns(
        bundle.ood_manifest,
        columns=("transcript_duration_s", "video_duration_s", "audio_duration_s"),
        manifest_name="ood_manifest",
    )
    if bundle.stimulus_manifest.duplicated(
        subset=["stimulus_namespace", "stimulus_id"]
    ).any():
        raise ValueError("stimulus_manifest contains duplicate stimulus rows")
    if bundle.friends_s7_manifest.duplicated(
        subset=["stimulus_namespace", "stimulus_id"]
    ).any():
        raise ValueError("friends_s7_manifest contains duplicate stimulus rows")
    if bundle.ood_manifest.duplicated(
        subset=["stimulus_namespace", "stimulus_id"]
    ).any():
        raise ValueError("ood_manifest contains duplicate stimulus rows")
    expected_hashes = build_manifest_hashes(
        run_manifest=bundle.run_manifest,
        stimulus_manifest=bundle.stimulus_manifest,
        friends_s7_manifest=bundle.friends_s7_manifest,
        ood_manifest=bundle.ood_manifest,
    )
    actual_hashes = bundle.metadata.get("hashes", {})
    if actual_hashes != expected_hashes:
        raise ValueError("manifest bundle hashes do not match manifest contents")


def load_manifest_bundle(bundle_dir: str | Path) -> ManifestBundle:
    bundle_path = Path(bundle_dir)
    paths = _bundle_paths(bundle_path)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Manifest bundle is incomplete at {bundle_path}. Missing: {', '.join(missing)}"
        )

    with open(paths["metadata"], encoding="utf-8") as f:
        metadata = json.load(f)

    run_manifest = load_tsv_with_schema(paths["run_manifest"], "run")
    stimulus_manifest = load_tsv_with_schema(paths["stimulus_manifest"], "stimulus")
    friends_s7_manifest = load_tsv_with_schema(paths["friends_s7_manifest"], "friends_s7")
    ood_manifest = load_tsv_with_schema(paths["ood_manifest"], "ood")

    bundle = ManifestBundle(
        bundle_dir=bundle_path,
        metadata=metadata,
        run_manifest=run_manifest,
        stimulus_manifest=stimulus_manifest,
        friends_s7_manifest=friends_s7_manifest,
        ood_manifest=ood_manifest,
    )
    validate_manifest_bundle(bundle)
    return bundle


def load_run_manifest_bundle(bundle_dir: str | Path) -> RunManifestBundle:
    bundle_path = Path(bundle_dir)
    paths = _bundle_paths(bundle_path)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Manifest bundle is incomplete at {bundle_path}. Missing: {', '.join(missing)}"
        )

    with open(paths["metadata"], encoding="utf-8") as f:
        metadata = json.load(f)

    run_manifest = load_tsv_with_schema(paths["run_manifest"], "run")
    validate_run_manifest_df(run_manifest, metadata=metadata)
    return RunManifestBundle(
        bundle_dir=bundle_path,
        metadata=metadata,
        run_manifest=run_manifest,
    )


def build_extraction_stimulus_manifest(bundle: ManifestBundle) -> pd.DataFrame:
    combined = pd.concat(
        [
            bundle.stimulus_manifest,
            bundle.friends_s7_manifest,
            bundle.ood_manifest,
        ],
        ignore_index=True,
        sort=False,
    )
    return combined.drop_duplicates(
        subset=["stimulus_namespace", "stimulus_id"]
    ).reset_index(drop=True)


def resolve_bundle_manifest_paths(
    df: pd.DataFrame,
    *,
    datapath: str | Path,
) -> pd.DataFrame:
    root = Path(datapath)
    out = df.copy()
    column_mapping = (
        ("fmri_relpath", "fmri_h5_path"),
        ("video_relpath", "video_path"),
        ("transcript_relpath", "transcript_path"),
        ("audio_relpath", "audio_path"),
    )
    for source_column, target_column in column_mapping:
        if source_column not in out.columns:
            continue
        out[target_column] = out[source_column].map(
            lambda value: "" if str(value or "").strip() == "" else str((root / str(value)).resolve())
        ).astype("string")
    return out


def save_training_split_artifacts(
    run_dir: str | Path,
    *,
    train_run_manifest: pd.DataFrame,
    val_run_manifest: pd.DataFrame,
    train_segment_manifest: pd.DataFrame,
    val_segment_manifest: pd.DataFrame,
) -> None:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    train_run_manifest.to_csv(run_path / "train_run_manifest.tsv", **_tsv_kwargs())
    val_run_manifest.to_csv(run_path / "val_run_manifest.tsv", **_tsv_kwargs())
    train_segment_manifest.to_csv(run_path / "train_segment_manifest.tsv", **_tsv_kwargs())
    val_segment_manifest.to_csv(run_path / "val_segment_manifest.tsv", **_tsv_kwargs())
