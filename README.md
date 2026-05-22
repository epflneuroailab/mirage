# MIRAGE: Adaptive Multimodal Gating for Whole-Brain fMRI Encoding

MIRAGE is a multimodal whole-brain fMRI encoder for naturalistic video. It uses
Qwen3-Omni hidden-state features from video, audio, and transcript streams, then
predicts BOLD responses in 1,000 cortical parcels for the Algonauts 2025
subjects.

This repository contains the training, evaluation, submission, and
video inference code for the MIRAGE preprint. Pretrained weights are
hosted on Hugging Face:

```text
https://huggingface.co/epfl-neuroai/mirage
```

## Install

```bash
git clone https://github.com/epflneuroailab/mirage
cd mirage

uv venv --python 3.12 .venv
uv pip install --python .venv -e .
source .venv/bin/activate
```

Set cluster-local paths with environment variables. Relative paths are resolved
under `SCRATCHPATH`.

```bash
export SCRATCHPATH=./scratch
export DATASET_PATH=datasets/algonauts_2025
export OUTPUT_PATH=outputs/mirage
```

## Weights

Download the public model files from Hugging Face:

```bash
hf download epfl-neuroai/mirage \
  model.safetensors config.yaml \
  --local-dir weights/mirage
```

Run fMRI inference for one video:

```bash
python -m brain_enc.cli.infer_fmri \
  --video /path/to/video.mp4 \
  --transcript /path/to/transcript.json \
  --run-dir weights/mirage \
  --subject-idx 0 \
  --output outputs/example_fmri.npy
```

`--subject-idx` uses the Algonauts subject order:
`0=sub-01`, `1=sub-02`, `2=sub-03`, `3=sub-05`.

## Online Video Demo

For a one-command test on a short online MP4, run:

```bash
bash scripts/run_online_video_demo.sh
```

The script downloads a demo video, downloads the public MIRAGE weights if
needed, runs fMRI inference for `sub-01`, and writes predicted fMRI plus
glass-brain PNG/MP4 visualizations under:

```text
outputs/online_video_demo/
```

Use `MIRAGE_VIDEO_URL`, `MIRAGE_SUBJECTS`, `MIRAGE_DEVICE`,
`MIRAGE_VIDEO_FPS`, and `MIRAGE_VIDEO_MAX_FRAMES` to customize the demo.

## Public Workflows

Training requires cached features, so run extraction first:

1. [Feature extraction](docs/feature_extraction.md) — required before training.
2. [Training](docs/training.md)
3. [Evaluation and S7/OOD submissions](docs/eval_submission.md)
4. [Parcel-weighted ensembling](docs/eval_submission.md#ensembling) — optional, combines multiple trained runs.

Manifest inference uses the downloaded Hugging Face weights and does not
require local feature extraction:

- [Manifest fMRI inference previews](docs/sample_fmri_visualization.md)

The selected public model config is:

```text
configs/experiments/mirage.yaml
```

## Results

MIRAGE results on the Algonauts 2025 CNeuroMod splits. Values are mean
Pearson r across the four trained subjects. Friends s06 is the held-out
validation split used during development; Friends s07 is the held-out
in-distribution benchmark; OOD is the held-out movie benchmark.

| Model | Friends s06 eval | Friends s07 held-out in-dist eval | OOD eval | Notes |
|---|---:|---:|---:|---|
| MIRAGE single model | 0.319 | 0.310 | 0.217 | Hugging Face checkpoint |
| MIRAGE 15-member ensemble | 0.335 | 0.323 | 0.227 | Algonauts 2025 final submission ensemble |

Per-subject Pearson r on the OOD test set:

| Subject | Pearson r |
|---|---|
| sub-01 | 0.244 |
| sub-02 | 0.210 |
| sub-03 | 0.235 |
| sub-05 | 0.179 |

## Citation

```bibtex
@article{gokce2026mirage,
  title = {MIRAGE: Adaptive Multimodal Gating for Whole-Brain fMRI Encoding},
  author = {Gokce, Abdulkadir and AlKhamissi, Badr and Schrimpf, Martin},
  journal = {arXiv preprint},
  year = {2026}
}
```
