# Manifest fMRI Inference Previews

## One-Command Online Demo

For a minimal public test, run:

```bash
bash scripts/run_online_video_demo.sh
```

The script downloads a short online MP4, downloads the public Hugging Face
checkpoint if needed, runs MIRAGE inference for `sub-01`, and writes:

- `online_sample_sub-01.npy`: predicted fMRI matrix with shape `(n_trs, 1000)`.
- `online_sample_sub-01.png`: compact matrix preview from manifest inference.
- `online_sample_sub-01_glass_mean.png`: mean predicted response as a glass brain.
- `online_sample_sub-01_glass_peak.png`: peak-TR predicted response as a glass brain.
- `online_sample_sub-01_glass_brain.mp4`: glass-brain video over predicted TRs.
- `online_sample_sub-01_glass_frames/`: rendered frames used for the MP4.

Useful overrides:

```bash
MIRAGE_VIDEO_URL=https://example.com/short.mp4 \
MIRAGE_SUBJECTS=sub-01 \
MIRAGE_DEVICE=cuda \
MIRAGE_VIDEO_FPS=8 \
MIRAGE_VIDEO_MAX_FRAMES=200 \
bash scripts/run_online_video_demo.sh
```

## Manifest Inference

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

The same command can be run through:

```bash
bash scripts/infer_fmri_manifest_example.sh
```

For each stimulus and subject, the command writes:

- `<stimulus>_<subject>.npy`: predicted fMRI matrix with shape `(n_trs, 1000)`.
- `<stimulus>_<subject>.png`: compact heatmap preview.
- `manifest_inference_summary.csv`: output index for downstream plotting.
- `manifest_inference_request.json`: reproducibility metadata.
