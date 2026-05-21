# Manifest fMRI Inference Previews

Use this workflow for small public video subsets such as later Koala-36M sample
manifests. The manifest must be CSV or TSV with these columns:

```text
stimulus_id,video_path,transcript_path
```

`transcript_path` is optional. Relative paths are resolved against the manifest
directory by default, or against `--path-root` when provided.

```bash
python -m brain_enc.cli.infer_fmri_manifest \
  --manifest samples/koala_subset.tsv \
  --path-root /path/to/koala_36m \
  --run-dir weights/mirage \
  --subjects sub-01,sub-02 \
  --output-dir outputs/koala_subset_fmri
```

For each stimulus and subject, the command writes:

- `<stimulus>_<subject>.npy`: predicted fMRI matrix with shape `(n_trs, 1000)`.
- `<stimulus>_<subject>.png`: compact heatmap preview.
- `manifest_inference_summary.csv`: output index for downstream plotting.
- `manifest_inference_request.json`: reproducibility metadata.
