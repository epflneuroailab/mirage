"""Parcel-weighted prediction ensembling."""


import json
import logging
import os
from pathlib import Path
import typing as tp
import zipfile

import numpy as np

logger = logging.getLogger(__name__)

_VALIDATION_ARTIFACTS = (
    "val_predictions.npy",
    "val_targets.npy",
    "val_subject_ids.npy",
)


def _progress(
    iterable: tp.Iterable[tp.Any],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    leave: bool = False,
) -> tp.Iterable[tp.Any]:
    """Wrap an iterable with tqdm unless progress has been disabled."""
    if os.environ.get("TQDM_DISABLE", "0") == "1":
        return iterable
    from tqdm.auto import tqdm

    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)


def infer_prediction_files_from_member_dirs(
    member_dirs: tp.Sequence[str | Path],
    *,
    benchmark: str = "friends_s7",
) -> list[Path]:
    """Resolve benchmark ``submission.npy`` files from run-local summaries."""
    prediction_files: list[Path] = []
    member_iter = _progress(
        member_dirs,
        desc="Resolve prediction files",
        total=len(member_dirs),
        unit="run",
    )
    for member_dir in member_iter:
        member_path = Path(member_dir)
        summary_path = member_path / "submission_artifacts.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing {summary_path}. Provide --prediction-files explicitly, "
                "or generate submissions for this run first."
            )

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        if not isinstance(summary, dict):
            raise ValueError(f"{summary_path} must contain a JSON object")
        if benchmark not in summary:
            available = ", ".join(sorted(str(key) for key in summary)) or "<none>"
            raise KeyError(
                f"{summary_path} has no benchmark {benchmark!r}. "
                f"Available benchmarks: {available}"
            )
        entry = summary[benchmark]
        if not isinstance(entry, dict) or not entry.get("submission_npy"):
            raise ValueError(
                f"{summary_path} benchmark {benchmark!r} does not record submission_npy"
            )
        prediction_path = Path(str(entry["submission_npy"]))
        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Resolved prediction file does not exist for {member_path}: "
                f"{prediction_path}"
            )
        prediction_files.append(prediction_path)
    return prediction_files


def _default_subject_names() -> list[str]:
    from brain_enc.data.algonauts import _SUBJECTS

    return sorted(_SUBJECTS)


def load_member_scores(member_dirs: tp.Sequence[str | Path]) -> np.ndarray:
    """Load ``pearson_per_parcel.npy`` from each member run directory."""
    scores: list[np.ndarray] = []
    member_iter = _progress(
        member_dirs,
        desc="Load parcel scores",
        total=len(member_dirs),
        unit="run",
    )
    for member_dir in member_iter:
        path = Path(member_dir) / "pearson_per_parcel.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing validation score artifact: {path}")
        scores.append(np.asarray(np.load(path), dtype=np.float32))
    if not scores:
        raise ValueError("At least one ensemble member is required")
    n_parcels = scores[0].shape
    for idx, score in enumerate(scores):
        if score.shape != n_parcels:
            raise ValueError(
                f"Member {idx} score shape {score.shape} does not match {n_parcels}"
            )
    return np.stack(scores, axis=0)


def member_has_validation_artifacts(member_dir: str | Path) -> bool:
    """Return whether a run directory has raw validation prediction artifacts."""
    member_path = Path(member_dir)
    return all((member_path / name).exists() for name in _VALIDATION_ARTIFACTS)


def _pearson_per_parcel(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return parcel-wise Pearson over rows, with NaN for degenerate columns."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"pred and target must be 2D arrays, got {pred.shape} and {target.shape}"
        )
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} does not match target {target.shape}")

    pred_z = pred - pred.mean(axis=0, keepdims=True)
    target_z = target - target.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(pred_z, axis=0) * np.linalg.norm(target_z, axis=0)
    out = np.full(pred.shape[1], np.nan, dtype=np.float32)
    valid = denom > 0.0
    out[valid] = (
        np.sum(pred_z[:, valid] * target_z[:, valid], axis=0) / denom[valid]
    ).astype(np.float32)
    return out


def load_member_subject_parcel_scores(
    member_dirs: tp.Sequence[str | Path],
    *,
    subject_names: tp.Sequence[str] | None = None,
) -> np.ndarray:
    """Compute validation Pearson shaped ``(models, subjects, parcels)``.

    Scores come from ``val_predictions.npy``, ``val_targets.npy``, and
    ``val_subject_ids.npy`` saved by training. Subject indices retain their
    training-time numeric IDs, so ``sub-05`` remains subject index 3 rather
    than being compacted to the third unique validation subject.
    """
    if not member_dirs:
        raise ValueError("At least one ensemble member is required")

    loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    n_subjects = len(subject_names or [])
    n_parcels: int | None = None

    member_iter = _progress(
        enumerate(member_dirs),
        desc="Load validation scores",
        total=len(member_dirs),
        unit="run",
    )
    for member_idx, member_dir in member_iter:
        member_path = Path(member_dir)
        missing = [name for name in _VALIDATION_ARTIFACTS if not (member_path / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing validation artifacts in {member_path}: {', '.join(missing)}"
            )

        preds = np.asarray(np.load(member_path / "val_predictions.npy"), dtype=np.float32)
        targets = np.asarray(np.load(member_path / "val_targets.npy"), dtype=np.float32)
        subject_ids = np.asarray(np.load(member_path / "val_subject_ids.npy"), dtype=np.int64)

        if preds.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                f"Member {member_idx} validation arrays must be 2D, got "
                f"{preds.shape} and {targets.shape}"
            )
        if preds.shape != targets.shape:
            raise ValueError(
                f"Member {member_idx} val_predictions shape {preds.shape} does not "
                f"match val_targets shape {targets.shape}"
            )
        if subject_ids.ndim != 1 or subject_ids.shape[0] != preds.shape[0]:
            raise ValueError(
                f"Member {member_idx} val_subject_ids shape {subject_ids.shape} does "
                f"not align with validation rows {preds.shape[0]}"
            )
        if subject_ids.size == 0:
            raise ValueError(f"Member {member_idx} validation subject IDs are empty")
        if np.any(subject_ids < 0):
            raise ValueError(f"Member {member_idx} validation subject IDs must be non-negative")
        if n_parcels is None:
            n_parcels = int(preds.shape[1])
        elif preds.shape[1] != n_parcels:
            raise ValueError(
                f"Member {member_idx} parcel count {preds.shape[1]} does not match "
                f"member 0 parcel count {n_parcels}"
            )
        n_subjects = max(n_subjects, int(subject_ids.max()) + 1)
        loaded.append((preds, targets, subject_ids))

    if n_parcels is None:
        raise ValueError("At least one ensemble member is required")

    scores = np.full((len(loaded), n_subjects, n_parcels), np.nan, dtype=np.float32)
    score_iter = _progress(
        enumerate(loaded),
        desc="Score validation parcels",
        total=len(loaded),
        unit="run",
    )
    for member_idx, (preds, targets, subject_ids) in score_iter:
        for subject_idx in np.unique(subject_ids):
            rows = subject_ids == subject_idx
            if int(rows.sum()) >= 2:
                scores[member_idx, int(subject_idx)] = _pearson_per_parcel(
                    preds[rows],
                    targets[rows],
                )
    return scores


def parcel_softmax_weights(
    scores: np.ndarray,
    *,
    temperature: float = 0.3,
) -> np.ndarray:
    """Return model weights shaped ``(n_models, n_parcels)``."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D (models, parcels), got {scores.shape}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    weights = np.zeros_like(scores, dtype=np.float32)
    for parcel_idx in range(scores.shape[1]):
        column = scores[:, parcel_idx]
        finite = np.isfinite(column)
        if not bool(finite.any()):
            weights[:, parcel_idx] = 1.0 / float(scores.shape[0])
            continue
        logits = np.full_like(column, -np.inf, dtype=np.float32)
        logits[finite] = column[finite] / float(temperature)
        logits[finite] -= np.max(logits[finite])
        exp = np.zeros_like(column, dtype=np.float32)
        exp[finite] = np.exp(logits[finite])
        weights[:, parcel_idx] = exp / np.sum(exp)
    return weights


def subject_parcel_softmax_weights(
    scores: np.ndarray,
    *,
    temperature: float = 0.3,
) -> np.ndarray:
    """Return model weights shaped ``(n_models, n_subjects, n_parcels)``."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 3:
        raise ValueError(
            f"scores must be 3D (models, subjects, parcels), got {scores.shape}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    weights = np.zeros_like(scores, dtype=np.float32)
    for subject_idx in range(scores.shape[1]):
        weights[:, subject_idx, :] = parcel_softmax_weights(
            scores[:, subject_idx, :],
            temperature=temperature,
        )
    return weights


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _load_aligned_validation_artifacts(
    member_dirs: tp.Sequence[str | Path],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Load raw validation artifacts, requiring identical row order."""
    predictions: list[np.ndarray] = []
    reference_targets: np.ndarray | None = None
    reference_subject_ids: np.ndarray | None = None

    member_iter = _progress(
        enumerate(member_dirs),
        desc="Load validation predictions",
        total=len(member_dirs),
        unit="run",
    )
    for member_idx, member_dir in member_iter:
        member_path = Path(member_dir)
        missing = [name for name in _VALIDATION_ARTIFACTS if not (member_path / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing validation artifacts in {member_path}: {', '.join(missing)}"
            )

        preds = np.asarray(np.load(member_path / "val_predictions.npy"), dtype=np.float32)
        targets = np.asarray(np.load(member_path / "val_targets.npy"), dtype=np.float32)
        subject_ids = np.asarray(np.load(member_path / "val_subject_ids.npy"), dtype=np.int64)

        if preds.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                f"Member {member_idx} validation arrays must be 2D, got "
                f"{preds.shape} and {targets.shape}"
            )
        if preds.shape != targets.shape:
            raise ValueError(
                f"Member {member_idx} val_predictions shape {preds.shape} does not "
                f"match val_targets shape {targets.shape}"
            )
        if subject_ids.ndim != 1 or subject_ids.shape[0] != preds.shape[0]:
            raise ValueError(
                f"Member {member_idx} val_subject_ids shape {subject_ids.shape} does "
                f"not align with validation rows {preds.shape[0]}"
            )

        if reference_targets is None:
            reference_targets = targets
            reference_subject_ids = subject_ids
        else:
            if targets.shape != reference_targets.shape:
                raise ValueError(
                    f"Member {member_idx} validation target shape {targets.shape} "
                    f"does not match member 0 shape {reference_targets.shape}"
                )
            if not np.array_equal(subject_ids, reference_subject_ids):
                raise ValueError(
                    f"Member {member_idx} validation subject IDs do not match member 0"
                )
            if not np.allclose(targets, reference_targets, equal_nan=True):
                raise ValueError(
                    f"Member {member_idx} validation targets do not match member 0"
                )
        predictions.append(preds)

    if reference_targets is None or reference_subject_ids is None:
        raise ValueError("At least one ensemble member is required")
    return predictions, reference_targets, reference_subject_ids


def evaluate_validation_ensemble(
    member_dirs: tp.Sequence[str | Path],
    weights: np.ndarray,
    *,
    subject_names: tp.Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, tp.Any]]:
    """Apply ensemble weights to saved validation predictions and score them."""
    if len(member_dirs) == 0:
        raise ValueError("At least one ensemble member is required")
    if weights.shape[0] != len(member_dirs):
        raise ValueError(
            f"weights first dimension {weights.shape[0]} does not match "
            f"{len(member_dirs)} member dirs"
        )
    if weights.ndim not in {2, 3}:
        raise ValueError(
            "weights must have shape (models, parcels) or "
            f"(models, subjects, parcels), got {weights.shape}"
        )

    predictions, targets, subject_ids = _load_aligned_validation_artifacts(member_dirs)
    n_rows, n_parcels = targets.shape
    if weights.shape[-1] != n_parcels:
        raise ValueError(
            f"weights parcel count {weights.shape[-1]} does not match "
            f"validation parcel count {n_parcels}"
        )

    combined = np.zeros_like(targets, dtype=np.float32)
    pred_iter = _progress(
        enumerate(predictions),
        desc="Blend validation predictions",
        total=len(predictions),
        unit="model",
    )
    for model_idx, preds in pred_iter:
        if weights.ndim == 2:
            combined += preds * weights[model_idx].reshape(1, -1)
            continue

        for subject_idx in np.unique(subject_ids):
            subject_int = int(subject_idx)
            if subject_int >= weights.shape[1]:
                resolved_subjects = list(subject_names or _default_subject_names())
                subject_label = (
                    resolved_subjects[subject_int]
                    if subject_int < len(resolved_subjects)
                    else str(subject_int)
                )
                raise ValueError(
                    f"No subject-specific validation weight for {subject_label!r} "
                    f"at index {subject_int}"
                )
            rows = subject_ids == subject_idx
            combined[rows] += preds[rows] * weights[model_idx, subject_int].reshape(1, -1)

    ensemble_pearson = _pearson_per_parcel(combined, targets)
    member_pearson = np.stack(
        [
            _pearson_per_parcel(preds, targets)
            for preds in _progress(
                predictions,
                desc="Score member validation",
                total=len(predictions),
                unit="model",
            )
        ],
        axis=0,
    )
    member_mean_pearson = [_finite_mean(row) for row in member_pearson]
    finite_member_means = [value for value in member_mean_pearson if value is not None]
    ensemble_mean = _finite_mean(ensemble_pearson)
    best_member_mean = max(finite_member_means) if finite_member_means else None

    per_subject_mean: dict[str, float | None] = {}
    resolved_subjects = list(subject_names or _default_subject_names())
    for subject_idx in np.unique(subject_ids):
        rows = subject_ids == subject_idx
        if int(rows.sum()) < 2:
            score = None
        else:
            score = _finite_mean(_pearson_per_parcel(combined[rows], targets[rows]))
        subject_int = int(subject_idx)
        subject_label = (
            resolved_subjects[subject_int]
            if subject_int < len(resolved_subjects)
            else str(subject_int)
        )
        per_subject_mean[subject_label] = score

    metrics = {
        "status": "ok",
        "n_validation_rows": int(n_rows),
        "n_parcels": int(n_parcels),
        "mean_pearson": ensemble_mean,
        "member_mean_pearson": member_mean_pearson,
        "best_member_mean_pearson": best_member_mean,
        "delta_vs_best_member": (
            None
            if ensemble_mean is None or best_member_mean is None
            else float(ensemble_mean - best_member_mean)
        ),
        "per_subject_mean_pearson": per_subject_mean,
    }
    return ensemble_pearson.astype(np.float32, copy=False), metrics


def load_submission_dict(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load a challenge submission dictionary from ``submission.npy``."""
    loaded = np.load(Path(path), allow_pickle=True)
    if not hasattr(loaded, "item"):
        raise ValueError(f"Submission file {path} did not contain a dictionary")
    submission = loaded.item()
    if not isinstance(submission, dict):
        raise ValueError(f"Submission file {path} did not contain a dictionary")
    return tp.cast(dict[str, dict[str, np.ndarray]], submission)


def _subject_to_index(subject: str, subject_names: tp.Sequence[str]) -> int:
    try:
        return list(subject_names).index(subject)
    except ValueError as exc:
        raise ValueError(
            f"Subject {subject!r} is not present in subject_names={list(subject_names)!r}"
        ) from exc


def combine_submission_dicts(
    submissions: tp.Sequence[dict[str, dict[str, np.ndarray]]],
    weights: np.ndarray,
    *,
    subject_names: tp.Sequence[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Combine submission dictionaries with global or subject-specific weights."""
    if len(submissions) == 0:
        raise ValueError("At least one submission is required")
    if weights.shape[0] != len(submissions):
        raise ValueError(
            f"weights first dimension {weights.shape[0]} does not match "
            f"{len(submissions)} submissions"
        )
    if weights.ndim not in {2, 3}:
        raise ValueError(
            "weights must have shape (models, parcels) or "
            f"(models, subjects, parcels), got {weights.shape}"
        )
    if weights.ndim == 3:
        subject_names = list(subject_names or _default_subject_names())
        if weights.shape[1] > len(subject_names):
            raise ValueError(
                f"weights subject dimension {weights.shape[1]} exceeds "
                f"subject_names length {len(subject_names)}"
            )

    reference = submissions[0]
    out: dict[str, dict[str, np.ndarray]] = {}
    total_chunks = sum(len(chunks) for chunks in reference.values())
    chunk_iter = _progress(
        (
            (subject, chunk_key, ref_array)
            for subject, chunks in reference.items()
            for chunk_key, ref_array in chunks.items()
        ),
        desc="Blend submission chunks",
        total=total_chunks,
        unit="chunk",
    )
    for subject, chunk_key, ref_array in chunk_iter:
        if subject not in out:
            out[subject] = {}
        subject_weights = weights
        if weights.ndim == 3:
            assert subject_names is not None
            subject_idx = _subject_to_index(subject, subject_names)
            if subject_idx >= weights.shape[1]:
                raise ValueError(
                    f"No subject-specific weights for {subject!r} at index {subject_idx}"
                )
            subject_weights = weights[:, subject_idx, :]
        ref = np.asarray(ref_array)
        if ref.ndim != 2:
            raise ValueError(
                f"Expected prediction array (time, parcels) for {subject}/{chunk_key}, "
                f"got {ref.shape}"
            )
        if ref.shape[1] != subject_weights.shape[1]:
            raise ValueError(
                f"Prediction parcel count {ref.shape[1]} for {subject}/{chunk_key} "
                f"does not match weights parcel count {subject_weights.shape[1]}"
            )
        combined = np.zeros(ref.shape, dtype=np.float32)
        for model_idx, submission in enumerate(submissions):
            if subject not in submission or chunk_key not in submission[subject]:
                raise ValueError(f"Missing {subject}/{chunk_key} from member {model_idx}")
            pred = np.asarray(submission[subject][chunk_key], dtype=np.float32)
            if pred.shape != ref.shape:
                raise ValueError(
                    f"Shape mismatch for {subject}/{chunk_key} in member {model_idx}: "
                    f"{pred.shape} != {ref.shape}"
                )
            combined += pred * subject_weights[model_idx].reshape(1, -1)
        out[subject][chunk_key] = combined
    return out


def _resolve_weighting_mode(
    member_dirs: tp.Sequence[str | Path],
    weighting: str,
) -> str:
    if weighting not in {"auto", "global_parcel", "subject_parcel"}:
        raise ValueError(
            "weighting must be 'auto', 'global_parcel', or "
            f"'subject_parcel', got {weighting!r}"
        )
    if weighting != "auto":
        return weighting
    if all(member_has_validation_artifacts(member_dir) for member_dir in member_dirs):
        return "subject_parcel"
    return "global_parcel"


def ensemble_submission_files(
    *,
    member_dirs: tp.Sequence[str | Path],
    prediction_files: tp.Sequence[str | Path],
    out_dir: str | Path,
    temperature: float = 0.3,
    weighting: tp.Literal["auto", "global_parcel", "subject_parcel"] = "auto",
    subject_names: tp.Sequence[str] | None = None,
) -> dict[str, tp.Any]:
    """Ensemble submission files using validation-score softmax weights."""
    if len(member_dirs) != len(prediction_files):
        raise ValueError(
            f"Expected one prediction file per member dir, got "
            f"{len(member_dirs)} member dirs and {len(prediction_files)} prediction files"
        )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting ensemble: members=%d weighting=%s temperature=%.4f out_dir=%s",
        len(member_dirs),
        weighting,
        float(temperature),
        out_path,
    )

    weighting_mode = _resolve_weighting_mode(member_dirs, weighting)
    logger.info("Resolved weighting mode: %s", weighting_mode)
    if weighting_mode == "subject_parcel":
        resolved_subject_names = list(subject_names or _default_subject_names())
        scores = load_member_subject_parcel_scores(
            member_dirs,
            subject_names=resolved_subject_names,
        )
        logger.info("Loaded validation scores with shape %s", scores.shape)
        weights = subject_parcel_softmax_weights(scores, temperature=temperature)
    else:
        resolved_subject_names = subject_names
        scores = load_member_scores(member_dirs)
        logger.info("Loaded validation scores with shape %s", scores.shape)
        weights = parcel_softmax_weights(scores, temperature=temperature)
    logger.info("Computed ensemble weights with shape %s", weights.shape)

    submission_iter = _progress(
        prediction_files,
        desc="Load submissions",
        total=len(prediction_files),
        unit="file",
    )
    submissions = [load_submission_dict(path) for path in submission_iter]
    logger.info("Loaded %d submission files", len(submissions))
    combined = combine_submission_dicts(
        submissions,
        weights,
        subject_names=resolved_subject_names,
    )
    logger.info("Submission predictions blended")

    np.save(out_path / "member_val_scores.npy", scores)
    np.save(out_path / "ensemble_weights.npy", weights)
    logger.info("Saved member_val_scores.npy and ensemble_weights.npy")
    validation_metrics: dict[str, tp.Any] | None = None
    if all(member_has_validation_artifacts(member_dir) for member_dir in member_dirs):
        try:
            validation_pearson, validation_metrics = evaluate_validation_ensemble(
                member_dirs,
                weights,
                subject_names=resolved_subject_names,
            )
            np.save(out_path / "validation_ensemble_pearson_per_parcel.npy", validation_pearson)
            with open(
                out_path / "validation_ensemble_metrics.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(validation_metrics, f, indent=2, sort_keys=True)
            mean_pearson = validation_metrics["mean_pearson"]
            best_member = validation_metrics["best_member_mean_pearson"]
            delta = validation_metrics["delta_vs_best_member"]
            if mean_pearson is None or best_member is None or delta is None:
                logger.info("Validation ensemble scoring finished, but Pearson was all-NaN.")
            else:
                logger.info(
                    "Validation ensemble mean Pearson: %.4f "
                    "(best member %.4f, delta %+0.4f)",
                    mean_pearson,
                    best_member,
                    delta,
                )
        except ValueError as exc:
            validation_metrics = {"status": "skipped", "reason": str(exc)}
            with open(
                out_path / "validation_ensemble_metrics.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(validation_metrics, f, indent=2, sort_keys=True)
            logger.warning("Validation ensemble scoring skipped: %s", exc)

    submission_path = out_path / "submission.npy"
    np.save(submission_path, combined)
    zip_path = out_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_path, arcname="submission.npy")
    logger.info("Saved ensemble submission: %s", zip_path)

    manifest = {
        "temperature": float(temperature),
        "member_dirs": [str(Path(path)) for path in member_dirs],
        "prediction_files": [str(Path(path)) for path in prediction_files],
        "weighting": weighting_mode,
        "n_members": len(member_dirs),
        "n_parcels": int(weights.shape[-1]),
        "n_subjects": int(weights.shape[1]) if weights.ndim == 3 else None,
        "submission_npy": str(submission_path),
        "submission_zip": str(zip_path),
        "validation_ensemble_metrics": validation_metrics,
    }
    with open(out_path / "ensemble_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    logger.info("Saved ensemble manifest: %s", out_path / "ensemble_manifest.json")
    return manifest
