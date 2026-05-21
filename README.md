# MIRAGE Multimodal Whole Brain Encoder

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
export SCRATCHPATH=/mnt/scratch
export DATASET_PATH=datasets/algonauts_2025
export OUTPUT_PATH=outputs/mirage
```

## Weights

Download the public model files from Hugging Face:

```bash
huggingface-cli download epfl-neuroai/mirage \
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

## Citation

```bibtex
@misc{mirage2026,
  title = {MIRAGE: Multimodal Inference for Representational Alignment and General Encoding},
  author = {EPFL NeuroAI Lab},
  year = {2026},
  url = {https://github.com/epflneuroailab/mirage}
}
```
