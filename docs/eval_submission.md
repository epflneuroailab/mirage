# Evaluation And S7/OOD Submissions

Evaluate a trained run:

```bash
python -m brain_enc.cli.evaluate \
  --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name>
```

Generate Friends S7 predictions:

```bash
python -m brain_enc.cli.evaluate \
  --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name> \
  --predict-s7
```

Generate OOD movie predictions:

```bash
python -m brain_enc.cli.evaluate \
  --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name> \
  --predict-ood
```

Package both public benchmark submissions:

```bash
python -m brain_enc.cli.make_submission \
  --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name>
```

Package one benchmark:

```bash
python -m brain_enc.cli.make_submission \
  --run-dir $SCRATCHPATH/$OUTPUT_PATH/runs/<run_name> \
  --benchmark ood
```

Submission `.npy` files should be written with NumPy `<2.0`, which is pinned in
this package.

## Ensembling

Combine submissions from multiple trained runs with parcel-wise softmax weights
derived from each member's validation Pearson:

```bash
python -m brain_enc.cli.ensemble_predictions \
  --members \
    $SCRATCHPATH/$OUTPUT_PATH/runs/<run_a> \
    $SCRATCHPATH/$OUTPUT_PATH/runs/<run_b> \
  --benchmark friends_s7 \
  --out-dir $SCRATCHPATH/$OUTPUT_PATH/ensembles/<name>
```

Each member directory must contain `submission_artifacts.json` (written by
`make_submission`) and `pearson_per_parcel.npy` (written by `train`). When all
members also have `val_predictions.npy`, `val_targets.npy`, and
`val_subject_ids.npy`, the CLI auto-upgrades to subject-and-parcel weights and
records the resulting validation Pearson next to the blended submission.

Useful flags:

- `--prediction-files`: pass `submission.npy` paths explicitly instead of
  resolving them via `submission_artifacts.json`.
- `--benchmark {friends_s7,ood}`: pick which benchmark to ensemble.
- `--temperature`: softmax temperature on member parcel scores (default `0.3`).
- `--weighting {auto,global_parcel,subject_parcel}`: force a weighting mode.

Outputs in `--out-dir` include `submission.npy`, `submission.zip`,
`ensemble_weights.npy`, `member_val_scores.npy`, `ensemble_manifest.json`, and
(when validation artifacts are present) `validation_ensemble_metrics.json`.
