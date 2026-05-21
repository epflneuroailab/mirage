#!/usr/bin/env bash
set -euo pipefail

: "${MIRAGE_WEIGHTS_DIR:?Set MIRAGE_WEIGHTS_DIR to a directory with config.yaml and model.safetensors}"
: "${MIRAGE_SAMPLE_MANIFEST:?Set MIRAGE_SAMPLE_MANIFEST to a CSV/TSV manifest}"
: "${MIRAGE_SAMPLE_OUT_DIR:?Set MIRAGE_SAMPLE_OUT_DIR to the output directory}"

args=(
  --manifest "$MIRAGE_SAMPLE_MANIFEST" \
  --run-dir "$MIRAGE_WEIGHTS_DIR" \
  --subjects "${MIRAGE_SAMPLE_SUBJECTS:-sub-01}" \
  --output-dir "$MIRAGE_SAMPLE_OUT_DIR"
)
if [[ -n "${MIRAGE_SAMPLE_PATH_ROOT:-}" ]]; then
  args+=(--path-root "$MIRAGE_SAMPLE_PATH_ROOT")
fi

python -m brain_enc.cli.infer_fmri_manifest "${args[@]}"
