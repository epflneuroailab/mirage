"""Reproduction benchmarking report.

Loads artifacts from a completed training run and generates a human-readable
reproduction report:

- overall mean validation Pearson
- per-subject mean Pearson
- per-parcel Pearson histogram summary
- comparison against a reference run (optional)
- acceptance check: within ``tolerance`` of reference

Designed to be called from ``cli/evaluate.py`` or standalone.
"""


import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_TOLERANCE = 0.02   # absolute mean Pearson gap allowed for acceptance


def load_run_artifacts(run_dir: str | Path) -> dict:
    """Load all available evaluation artifacts from *run_dir*.

    Returns
    -------
    dict with keys:
        metrics           — trainer callback metrics (metrics.json)
        pearson_per_parcel — (n_parcels,) array
        pearson_per_subject — dict subject_id → mean Pearson (may be empty)
        config            — raw experiment config dict
    """
    run_dir = Path(run_dir)
    artifacts: dict = {}

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            artifacts["metrics"] = json.load(f)
    else:
        artifacts["metrics"] = {}
        logger.warning("metrics.json not found in %s", run_dir)

    parcel_path = run_dir / "pearson_per_parcel.npy"
    if parcel_path.exists():
        artifacts["pearson_per_parcel"] = np.load(parcel_path)
    else:
        artifacts["pearson_per_parcel"] = None
        logger.warning("pearson_per_parcel.npy not found in %s", run_dir)

    subj_path = run_dir / "pearson_per_subject.json"
    if subj_path.exists():
        with open(subj_path) as f:
            artifacts["pearson_per_subject"] = json.load(f)
    else:
        artifacts["pearson_per_subject"] = {}
        logger.warning("pearson_per_subject.json not found in %s", run_dir)

    config_path = run_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            artifacts["config"] = yaml.safe_load(f)
    else:
        artifacts["config"] = {}

    return artifacts


def generate_report(
    run_dir: str | Path,
    reference_mean_pearson: float | None = None,
    tolerance: float = _DEFAULT_TOLERANCE,
    save: bool = True,
) -> dict:
    """Generate a Phase 1 reproduction report.

    Parameters
    ----------
    run_dir:
        Directory of the completed training run.
    reference_mean_pearson:
        Optional reference value to compare against (e.g. from a reference run).
        If provided, an acceptance verdict is included in the report.
    tolerance:
        Allowed absolute gap from the reference for acceptance.
    save:
        Whether to write ``reproduction_report.json`` into *run_dir*.

    Returns
    -------
    Report dict.
    """
    run_dir = Path(run_dir)
    art = load_run_artifacts(run_dir)
    benchmark_cfg = art["config"].get("benchmark") or {}

    if reference_mean_pearson is None:
        reference_mean_pearson = benchmark_cfg.get("reference_mean_pearson")
    if tolerance == _DEFAULT_TOLERANCE:
        tolerance = benchmark_cfg.get("tolerance", tolerance)

    # ------------------------------------------------------------------
    # Overall Pearson
    # ------------------------------------------------------------------
    per_parcel = art["pearson_per_parcel"]
    if per_parcel is not None:
        mean_pearson = float(per_parcel.mean())
        median_pearson = float(np.median(per_parcel))
        p10 = float(np.percentile(per_parcel, 10))
        p90 = float(np.percentile(per_parcel, 90))
        n_positive = int((per_parcel > 0).sum())
    else:
        # Fall back to logged metric
        mean_pearson = art["metrics"].get("val/pearson", float("nan"))
        median_pearson = float("nan")
        p10 = p90 = float("nan")
        n_positive = -1

    # ------------------------------------------------------------------
    # Per-subject
    # ------------------------------------------------------------------
    per_subject = art["pearson_per_subject"]
    subject_mean = (
        float(np.mean(list(per_subject.values()))) if per_subject else float("nan")
    )

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------
    acceptance: dict = {}
    if reference_mean_pearson is not None:
        gap = abs(mean_pearson - reference_mean_pearson)
        accepted = gap <= tolerance
        acceptance = {
            "reference_mean_pearson": reference_mean_pearson,
            "achieved_mean_pearson": mean_pearson,
            "gap": round(gap, 4),
            "tolerance": tolerance,
            "accepted": accepted,
            "verdict": "PASS" if accepted else "FAIL",
        }
        logger.info(
            "Acceptance check: achieved=%.4f  ref=%.4f  gap=%.4f  %s",
            mean_pearson,
            reference_mean_pearson,
            gap,
            acceptance["verdict"],
        )
    else:
        acceptance = {
            "verdict": "SKIPPED",
            "reason": (
                "No reference_mean_pearson provided. Set benchmark.reference_mean_pearson "
                "in the saved config or pass --reference-pearson to evaluate parity."
            ),
            "tolerance": tolerance,
        }
        logger.info("Acceptance check skipped: no reference_mean_pearson provided.")

    report = {
        "run_dir": str(run_dir),
        "run_name": art["config"].get("run_name", ""),
        "overall": {
            "mean_pearson": round(mean_pearson, 4),
            "median_pearson": round(median_pearson, 4),
            "p10_pearson": round(p10, 4),
            "p90_pearson": round(p90, 4),
            "n_parcels_positive": n_positive,
        },
        "per_subject": {
            k: round(v, 4) for k, v in per_subject.items()
        },
        "subject_mean_pearson": round(subject_mean, 4),
        "trainer_metrics": art["metrics"],
        "acceptance": acceptance,
    }

    if save:
        report_path = run_dir / "reproduction_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Report saved to %s", report_path)

    return report


def inspect_feature_cache(
    dataset_name: str,
    extractor_map: dict[str, str | dict[str, object]],
    cache_dir: str | None = None,
    out_path: str | Path | None = None,
    output_suffix: str | None = None,
) -> dict:
    """Scan HDF5 feature stores and return a cache inspection summary.

    Parameters
    ----------
    dataset_name:
        Dataset identifier (e.g. ``"algonauts2025"``).
    extractor_map:
        Mapping of modality → extractor_id, e.g.
        ``{"text": "llama3p2", "audio": "wav2vecbert", "vision": "vjepa2",
           "fmri": "algonauts2025"}``.
    cache_dir:
        Override the default cache root (matches ``data.hdf5_cache_dir``).
    out_path:
        If given, write the summary JSON to this path.

    Returns
    -------
    dict with per-modality entries: path, exists, n_keys, file_size_mb.
    """
    from brain_enc.paths import resolve_feature_store_path

    summary: dict = {"dataset_name": dataset_name, "modalities": {}}

    for modality, extractor_entry in extractor_map.items():
        if isinstance(extractor_entry, dict):
            extractor_id = str(extractor_entry["extractor_id"])
            available_modalities = extractor_entry.get("available_modalities")
            stream_kind = extractor_entry.get("stream_kind")
            cache_variant = extractor_entry.get("cache_variant")
            prompt_id = extractor_entry.get("prompt_id")
        else:
            extractor_id = extractor_entry
            available_modalities = None
            stream_kind = None
            cache_variant = None
            prompt_id = None
        store_path = resolve_feature_store_path(
            dataset_name,
            modality,
            extractor_id,
            available_modalities=available_modalities,
            stream_kind=stream_kind,
            cache_variant=cache_variant,
            prompt_id=prompt_id,
            cache_dir=cache_dir,
            output_suffix=output_suffix,
        )
        entry: dict = {
            "path": str(store_path),
            "exists": store_path.exists(),
            "n_keys": 0,
            "file_size_mb": 0.0,
            "conditioning_id": None,
            "stream_kind": stream_kind,
            "prompt_id": prompt_id,
        }
        if available_modalities is not None:
            from brain_enc.modalities import conditioning_id

            entry["conditioning_id"] = conditioning_id(available_modalities, target_modality=modality)
        if store_path.exists():
            entry["file_size_mb"] = round(store_path.stat().st_size / (1024 ** 2), 2)
            try:
                import h5py
                # Walk the full HDF5 hierarchy — keys like "friends/s01e01a" are
                # stored as nested groups, so len(f.keys()) only counts top-level
                # groups and would undercount.
                with h5py.File(store_path, "r") as f:
                    item_names: list[str] = []

                    def _visit(name: str, obj: h5py.HLObject) -> None:
                        if isinstance(obj, h5py.Group) and "features" in obj:
                            item_names.append(name)

                    f.visititems(_visit)
                    entry["n_keys"] = len(item_names)
            except Exception as exc:
                entry["error"] = str(exc)
        summary["modalities"][modality] = entry
        status = f"{entry['n_keys']} keys, {entry['file_size_mb']} MB" if entry["exists"] else "missing"
        logger.info("Feature cache [%s/%s]: %s", modality, extractor_id, status)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Feature cache summary saved to %s", out_path)

    return summary


def print_report(report: dict) -> None:
    """Print a human-readable summary of the report to stdout."""
    print("\n" + "=" * 60)
    print(f"  Reproduction Report: {report.get('run_name', 'unknown run')}")
    print("=" * 60)
    ov = report["overall"]
    print(f"  Mean Pearson (parcels):  {ov['mean_pearson']:.4f}")
    print(f"  Median Pearson:          {ov['median_pearson']:.4f}")
    print(f"  P10 / P90:               {ov['p10_pearson']:.4f} / {ov['p90_pearson']:.4f}")
    print(f"  Parcels with r > 0:      {ov['n_parcels_positive']}")
    print(f"  Subject-mean Pearson:    {report['subject_mean_pearson']:.4f}")
    if report["per_subject"]:
        print("\n  Per-subject:")
        for sid, r in sorted(report["per_subject"].items()):
            print(f"    Subject {sid}: {r:.4f}")
    if report.get("acceptance"):
        acc = report["acceptance"]
        print(f"\n  Acceptance [{acc['verdict']}]")
        if acc["verdict"] == "SKIPPED":
            print(f"    {acc['reason']}")
        else:
            print(f"    Reference: {acc['reference_mean_pearson']:.4f}")
            print(f"    Achieved:  {acc['achieved_mean_pearson']:.4f}")
            print(f"    Gap:       {acc['gap']:.4f}  (tolerance <= {acc['tolerance']})")
    print("=" * 60 + "\n")
