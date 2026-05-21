"""CLI entry point: evaluate a trained run and generate a reproduction report.

Usage
-----
    # Generate report from a finished run
    python -m brain_enc.cli.evaluate --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name>

    # Compare against a known reference Pearson value
    python -m brain_enc.cli.evaluate \\
        --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name> \\
        --reference-pearson 0.312

    # Without a reference value, acceptance is reported as SKIPPED
    python -m brain_enc.cli.evaluate \\
        --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name>

    # Also generate Friends S7 predictions
    python -m brain_enc.cli.evaluate \\
        --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name> \\
        --predict-s7

    # Also generate OOD predictions
    python -m brain_enc.cli.evaluate \\
        --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name> \\
        --predict-ood
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained brain encoder run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a completed training run directory.",
    )
    parser.add_argument(
        "--reference-pearson",
        type=float,
        default=None,
        help=(
            "Optional reference mean Pearson for acceptance check (e.g. 0.312). "
            "If omitted and not set in config, acceptance is reported as SKIPPED."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Absolute Pearson tolerance for acceptance (default: 0.02).",
    )
    parser.add_argument(
        "--predict-s7",
        action="store_true",
        default=False,
        help="Also generate Friends S7 predictions.",
    )
    parser.add_argument(
        "--predict-ood",
        action="store_true",
        default=False,
        help="Also generate OOD movie predictions.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subjects to predict for (default: all four challenge subjects).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference windows per forward pass when generating submissions.",
    )
    parser.add_argument(
        "--datapath",
        default=None,
        help="Override the resolved dataset root for this command.",
    )
    args = parser.parse_args(argv)

    from pathlib import Path
    from brain_enc.eval.benchmark import generate_report, print_report

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error("Run dir not found: %s", run_dir)
        sys.exit(1)

    report = generate_report(
        run_dir=run_dir,
        reference_mean_pearson=args.reference_pearson,
        tolerance=args.tolerance,
        save=True,
    )
    print_report(report)

    if args.predict_s7 or args.predict_ood:
        from brain_enc.env import load_env
        load_env()
        from brain_enc.paths import submission_dir

        run_name = report.get("run_name") or run_dir.name
        if args.predict_s7:
            from brain_enc.eval.predict_friends_s7 import predict_friends_s7

            out_dir = submission_dir(run_name) / "friends_s7"
            logger.info("Generating Friends S7 predictions → %s", out_dir)
            saved = predict_friends_s7(
                run_dir=run_dir,
                out_dir=out_dir,
                datapath=args.datapath,
                subjects=args.subjects,
                batch_size=args.batch_size,
            )
            logger.info("S7 predictions written: %s", saved)
        if args.predict_ood:
            from brain_enc.eval.predict_ood import predict_ood

            out_dir = submission_dir(run_name) / "ood"
            logger.info("Generating OOD predictions → %s", out_dir)
            saved = predict_ood(
                run_dir=run_dir,
                out_dir=out_dir,
                datapath=args.datapath,
                subjects=args.subjects,
                batch_size=args.batch_size,
            )
            logger.info("OOD predictions written: %s", saved)


if __name__ == "__main__":
    main()
