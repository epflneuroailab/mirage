# Feature Extraction

Extract frozen stimulus features into HDF5 caches.

The flagship config extracts Qwen3-Omni post-fusion hidden states (no
extraction-time layer pooling) for all three modalities:

```bash
python -m brain_enc.cli.prepare_manifest \
  --config configs/experiments/mirage.yaml
```

```bash
python -m brain_enc.cli.extract_features \
  --config configs/experiments/mirage.yaml
```

Useful flags:

- `--stimulus-index N`: extract features for one stimulus.
- `--overwrite`: rewrite existing cache entries.
- `--save-dtype fp16|fp32`: choose stored feature dtype.

Feature caches are resolved from `SCRATCHPATH`, `DATASET_PATH`, and the config.