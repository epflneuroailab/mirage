"""Batch raw-video fMRI inference from a small public manifest."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brain_enc.eval.infer_fmri import InferenceRequest, SUBJECTS, run_inference
from brain_enc.eval.model_loading import default_checkpoint


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return slug or "stimulus"


def _read_manifest(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(path, sep=sep)
    missing = {"stimulus_id", "video_path"} - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return df


def _resolve_path(value: object, *, root: Path | None, manifest_dir: Path) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    base = root if root is not None else manifest_dir
    return (base / path).resolve()


def _parse_subjects(value: str) -> list[int]:
    subjects: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if item in SUBJECTS:
            subjects.append(SUBJECTS.index(item))
            continue
        idx = int(item)
        if idx < 0 or idx >= len(SUBJECTS):
            raise ValueError(f"Subject index must be in [0, {len(SUBJECTS) - 1}]: {idx}")
        subjects.append(idx)
    if not subjects:
        raise ValueError("At least one subject must be requested.")
    return subjects


def _write_preview(predictions: np.ndarray, output_path: Path) -> None:
    """Write a compact heatmap preview for a predicted fMRI matrix."""

    import matplotlib.pyplot as plt

    n_parcels = predictions.shape[1]
    parcel_step = max(1, n_parcels // 250)
    preview = predictions[:, ::parcel_step]

    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    image = ax.imshow(preview.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
    ax.set_xlabel("TR")
    ax.set_ylabel("Parcels")
    fig.colorbar(image, ax=ax, label="Predicted response")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_manifest_inference(
    *,
    manifest_path: Path,
    output_dir: Path,
    run_dir: Path,
    checkpoint: Path,
    subjects: list[int],
    device: str,
    batch_size: int,
    path_root: Path | None = None,
    make_previews: bool = True,
) -> list[dict[str, object]]:
    """Run MIRAGE inference for each manifest row and requested subject."""

    df = _read_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        stimulus_id = str(row["stimulus_id"])
        video = _resolve_path(row["video_path"], root=path_root, manifest_dir=manifest_path.parent)
        transcript = _resolve_path(
            row.get("transcript_path"),
            root=path_root,
            manifest_dir=manifest_path.parent,
        )
        if video is None or not video.exists():
            raise FileNotFoundError(f"Video for {stimulus_id!r} does not exist: {video}")
        if transcript is not None and not transcript.exists():
            raise FileNotFoundError(
                f"Transcript for {stimulus_id!r} does not exist: {transcript}"
            )

        for subject_idx in subjects:
            subject = SUBJECTS[subject_idx]
            stem = f"{_slug(stimulus_id)}_{subject}"
            prediction_path = output_dir / f"{stem}.npy"
            preview_path = output_dir / f"{stem}.png"
            request = InferenceRequest(
                video=video,
                transcript=transcript,
                run_dir=run_dir,
                checkpoint=checkpoint,
                subject_idx=subject_idx,
                output=prediction_path,
                device=device,
                batch_size=batch_size,
            )
            logger.info("Running %s for %s", stimulus_id, subject)
            predictions = run_inference(request)
            np.save(prediction_path, predictions.astype(np.float32, copy=False))
            if make_previews:
                _write_preview(predictions, preview_path)
            rows.append(
                {
                    "stimulus_id": stimulus_id,
                    "subject": subject,
                    "video_path": str(video),
                    "transcript_path": "" if transcript is None else str(transcript),
                    "prediction_path": str(prediction_path),
                    "preview_path": str(preview_path) if make_previews else "",
                    "n_trs": int(predictions.shape[0]),
                    "n_parcels": int(predictions.shape[1]),
                }
            )

    summary_path = output_dir / "manifest_inference_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "manifest_inference_request.json").open("w") as f:
        json.dump(
            {
                "manifest_path": str(manifest_path),
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "subjects": [SUBJECTS[idx] for idx in subjects],
                "device": device,
                "batch_size": batch_size,
                "path_root": None if path_root is None else str(path_root),
                "make_previews": make_previews,
            },
            f,
            indent=2,
        )
    logger.info("Wrote summary: %s", summary_path)
    return rows


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MIRAGE fMRI inference over a small CSV/TSV manifest with "
            "stimulus_id, video_path, and optional transcript_path columns."
        )
    )
    parser.add_argument("--manifest", required=True, help="CSV/TSV sample manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for .npy and preview files.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run/HF directory with config.yaml and model weights.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional weights path. Defaults to model.safetensors, best.ckpt, or last.ckpt.",
    )
    parser.add_argument(
        "--subjects",
        default="sub-01",
        help="Comma-separated subject names or zero-based indices. Example: sub-01,sub-02",
    )
    parser.add_argument(
        "--path-root",
        default=None,
        help="Optional root for relative video/transcript paths. Defaults to manifest directory.",
    )
    parser.add_argument("--device", default=None, help="Torch device. Defaults to CUDA when available.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-previews", action="store_true", help="Skip PNG heatmap previews.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint is not None
        else default_checkpoint(run_dir)
    )
    if checkpoint is None:
        raise FileNotFoundError(
            f"No model.safetensors, best.ckpt, or last.ckpt found under {run_dir}"
        )
    path_root = Path(args.path_root).expanduser().resolve() if args.path_root else None
    rows = run_manifest_inference(
        manifest_path=manifest_path,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        run_dir=run_dir,
        checkpoint=checkpoint,
        subjects=_parse_subjects(args.subjects),
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=args.batch_size,
        path_root=path_root,
        make_previews=not args.no_previews,
    )
    logger.info("Finished %d prediction files.", len(rows))


if __name__ == "__main__":
    main()
