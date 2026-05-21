"""Friends Season 7 prediction generator.

Thin wrapper around the generic challenge submission predictor for the
in-distribution Friends S7 benchmark.
"""


from pathlib import Path

from brain_enc.eval.predict_submission import (
    _write_submission_sidecar_artifacts,
    predict_submission_benchmark,
    resolve_sample_count_file,
)


def _resolve_s7_sample_count_file(
    datapath: str | Path,
    subject: str,
    repo_root: str | Path | None = None,
) -> Path | None:
    """Backward-compatible Friends S7 helper for sample-count lookup."""
    return resolve_sample_count_file(
        datapath=datapath,
        subject=subject,
        benchmark="friends_s7",
        repo_root=repo_root,
    )


def predict_friends_s7(
    run_dir,
    out_dir,
    datapath=None,
    subjects=None,
    batch_size: int = 8,
    device: str | None = None,
    prediction_mode: str = "default",
):
    return predict_submission_benchmark(
        benchmark="friends_s7",
        run_dir=run_dir,
        out_dir=out_dir,
        datapath=datapath,
        subjects=subjects,
        batch_size=batch_size,
        device=device,
        prediction_mode=prediction_mode,
    )
