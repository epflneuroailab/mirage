# Training

Train MIRAGE from cached HDF5 features.

Prerequisite: run [feature extraction](feature_extraction.md) first so the
HDF5 caches exist under `$SCRATCHPATH/$DATASET_PATH/extracted_features/`.

```bash
MIRAGE_CFG=configs/experiments/mirage.yaml

python -m brain_enc.cli.train \
  --config "$MIRAGE_CFG"
```

Common overrides:

```bash
python -m brain_enc.cli.train \
  --config "$MIRAGE_CFG" \
  training.n_epochs=5 \
  optim.optimizer.lr=3e-4
```

Restrict stimulus modalities:

```bash
python -m brain_enc.cli.train \
  --config "$MIRAGE_CFG" \
  data.modalities=text,audio
```

Typical run outputs:

```text
$SCRATCHPATH/$OUTPUT_PATH/runs/<run_name>/
├── config.yaml
├── metrics.json
├── pearson_per_parcel.npy
├── pearson_per_subject.json
├── best.ckpt
└── last.ckpt
```
