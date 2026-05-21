"""CLI entry point for parcel-weighted ensembling."""


import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble submission.npy files with parcel-wise validation weights."
    )
    parser.add_argument(
        "--members",
        nargs="+",
        required=True,
        help=(
            "Run directories containing validation artifacts. With --weighting=auto, "
            "val_predictions/val_targets/val_subject_ids are preferred when present; "
            "otherwise pearson_per_parcel.npy is used."
        ),
    )
    parser.add_argument(
        "--prediction-files",
        nargs="+",
        default=None,
        help="submission.npy files to ensemble, in the same order as --members.",
    )
    parser.add_argument(
        "--benchmark",
        default="friends_s7",
        choices=("friends_s7", "ood"),
        help=(
            "Benchmark prediction to infer from each member's "
            "submission_artifacts.json when --prediction-files is omitted."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for ensembled submission.npy and metadata.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Softmax temperature for parcel-wise model weights.",
    )
    parser.add_argument(
        "--weighting",
        choices=("auto", "global_parcel", "subject_parcel"),
        default="auto",
        help=(
            "How to derive ensemble weights. auto uses per-subject/per-parcel "
            "weights when raw validation artifacts exist, else global per-parcel."
        ),
    )
    args = parser.parse_args(argv)

    from brain_enc.eval.ensemble_predictions import (
        ensemble_submission_files,
        infer_prediction_files_from_member_dirs,
    )

    prediction_files = args.prediction_files
    if prediction_files is None:
        prediction_files = infer_prediction_files_from_member_dirs(
            args.members,
            benchmark=args.benchmark,
        )

    manifest = ensemble_submission_files(
        member_dirs=args.members,
        prediction_files=prediction_files,
        out_dir=args.out_dir,
        temperature=args.temperature,
        weighting=args.weighting,
    )
    logger.info("Ensemble written: %s", manifest)


if __name__ == "__main__":
    main()
