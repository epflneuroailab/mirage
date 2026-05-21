"""OOD movie prediction generator.

Thin wrapper around the generic challenge submission predictor for the
out-of-distribution benchmark.
"""


from brain_enc.eval.predict_submission import predict_submission_benchmark


def predict_ood(
    run_dir,
    out_dir,
    datapath=None,
    subjects=None,
    batch_size: int = 8,
    device: str | None = None,
    prediction_mode: str = "default",
):
    return predict_submission_benchmark(
        benchmark="ood",
        run_dir=run_dir,
        out_dir=out_dir,
        datapath=datapath,
        subjects=subjects,
        batch_size=batch_size,
        device=device,
        prediction_mode=prediction_mode,
    )
