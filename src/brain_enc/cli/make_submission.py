"""CLI entry point: generate challenge-format submission bundles.

Produces per-subject prediction files for the public held-out benchmarks under
the configured submission directory, then prints a summary.

Usage
-----
    # Generate both public benchmark submissions from best checkpoint
    python -m brain_enc.cli.make_submission \\
        --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name>

    # OOD only
    python -m brain_enc.cli.make_submission \\
        --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name> \\
        --benchmark ood

    # Specific subjects only
    python -m brain_enc.cli.make_submission \\
        --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name> \\
        --benchmark friends_s7 \\
        --subjects sub-01 sub-02

    # Override base output directory
    python -m brain_enc.cli.make_submission \\
        --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name> \\
        --out-dir /path/to/submission
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Package a trained run into a challenge submission."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a completed run directory with model.safetensors or best.ckpt.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Where to write submission files. For --benchmark all this is treated as "
            "the base directory and child dirs are created per benchmark. Defaults to "
            "$SCRATCHPATH/$OUTPUT_PATH/submissions/<run_name>/."
        ),
    )
    parser.add_argument(
        "--benchmark",
        choices=["all", "friends_s7", "id_dist", "ood"],
        default="all",
        help="Which public benchmark submission to generate (default: all).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subjects to predict for (default: sub-01 sub-02 sub-03 sub-05).",
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
    parser.add_argument(
        "--prediction-mode",
        choices=["default", "group_only", "subject_mean"],
        default="default",
        help=(
            "Submission prediction mode. 'group_only' exports the shared group branch "
            "for group_residual_subject checkpoints. 'subject_mean' averages default "
            "subject-specific predictions and reuses that average for every subject."
        ),
    )
    args = parser.parse_args(argv)

    from brain_enc.env import load_env
    load_env()
    from brain_enc.eval.predict_submission import (
        format_submission_results,
        generate_submission_artifacts,
    )

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error("Run dir not found: %s", run_dir)
        sys.exit(1)

    from brain_enc.eval.model_loading import default_checkpoint

    checkpoint = default_checkpoint(run_dir)
    if checkpoint is None:
        logger.error(
            "No model.safetensors, best.ckpt, or last.ckpt in %s — train first or check run_dir.",
            run_dir,
        )
        sys.exit(1)

    results = generate_submission_artifacts(
        run_dir=run_dir,
        out_dir=args.out_dir,
        benchmark=args.benchmark,
        subjects=args.subjects,
        batch_size=args.batch_size,
        datapath=args.datapath,
        prediction_mode=args.prediction_mode,
    )
    print(format_submission_results(results))


if __name__ == "__main__":
    main()
