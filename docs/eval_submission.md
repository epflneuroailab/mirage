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
