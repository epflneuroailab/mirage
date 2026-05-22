#!/usr/bin/env bash
set -euo pipefail

# One-command public MIRAGE demo:
#   1. download a short online MP4
#   2. download/use public MIRAGE weights from Hugging Face
#   3. run fMRI inference
#   4. write glass-brain visualizations from Schaefer1000 parcel predictions
#
# Run from the MIRAGE repository root:
#   bash scripts/run_online_video_demo.sh
#
# Useful overrides:
#   MIRAGE_VIDEO_URL=https://.../clip.mp4
#   MIRAGE_SUBJECTS=sub-01
#   MIRAGE_DEVICE=cuda
#   MIRAGE_BATCH_SIZE=1
#   MIRAGE_VIDEO_FPS=6
#   MIRAGE_VIDEO_MAX_FRAMES=100
#   MIRAGE_HF_REPO=epfl-neuroai/mirage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DOTENV_PATH="${DOTENV_PATH:-${REPO_ROOT}/.env}"
if [[ -f "${DOTENV_PATH}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${DOTENV_PATH}"
  set +a
fi

PYTHON_BIN="${MIRAGE_PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/online_video_demo}"
WORK_DIR="${WORK_DIR:-${RUN_ROOT}/work}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/predictions}"
WEIGHTS_DIR="${MIRAGE_WEIGHTS_DIR:-${WORK_DIR}/weights/mirage}"
VIDEO_URL="${MIRAGE_VIDEO_URL:-https://raw.githubusercontent.com/bower-media-samples/big-buck-bunny-1080p-30s/master/video.mp4}"
VIDEO_PATH="${VIDEO_PATH:-${WORK_DIR}/online_sample.mp4}"
MANIFEST_PATH="${MANIFEST_PATH:-${WORK_DIR}/online_sample.tsv}"

MIRAGE_HF_REPO="${MIRAGE_HF_REPO:-epfl-neuroai/mirage}"
MIRAGE_SUBJECTS="${MIRAGE_SUBJECTS:-sub-01}"
MIRAGE_BATCH_SIZE="${MIRAGE_BATCH_SIZE:-1}"

mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}" "${WEIGHTS_DIR}"

# Keep downloads/cache local to the demo unless the caller already configured them.
export HF_HOME="${HF_HOME:-${WORK_DIR}/hf_home}"
export TORCH_HOME="${TORCH_HOME:-${WORK_DIR}/torch_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_DIR}/xdg_cache}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

echo "Downloading demo video:"
echo "  ${VIDEO_URL}"
VIDEO_URL="${VIDEO_URL}" VIDEO_PATH="${VIDEO_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
import shutil
import urllib.request
from pathlib import Path

url = os.environ["VIDEO_URL"]
path = Path(os.environ["VIDEO_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as f:
    shutil.copyfileobj(response, f)
print(f"Wrote {path} ({path.stat().st_size / 1024 / 1024:.2f} MB)")
PY

cat > "${MANIFEST_PATH}" <<EOF
stimulus_id	video_path	transcript_path
online_sample	${VIDEO_PATH}	
EOF

if [[ ! -f "${WEIGHTS_DIR}/config.yaml" || ! -f "${WEIGHTS_DIR}/model.safetensors" ]]; then
  echo "Downloading MIRAGE weights from Hugging Face:"
  echo "  ${MIRAGE_HF_REPO}"
  MIRAGE_HF_REPO="${MIRAGE_HF_REPO}" WEIGHTS_DIR="${WEIGHTS_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = os.environ["MIRAGE_HF_REPO"]
weights_dir = Path(os.environ["WEIGHTS_DIR"])
token = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACE_TOKEN")
)
weights_dir.mkdir(parents=True, exist_ok=True)
for filename in ("model.safetensors", "config.yaml"):
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=weights_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
PY
else
  echo "Using existing MIRAGE weights in ${WEIGHTS_DIR}"
fi

args=(
  --manifest "${MANIFEST_PATH}"
  --run-dir "${WEIGHTS_DIR}"
  --subjects "${MIRAGE_SUBJECTS}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${MIRAGE_BATCH_SIZE}"
)
if [[ -n "${MIRAGE_DEVICE:-}" ]]; then
  args+=(--device "${MIRAGE_DEVICE}")
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m brain_enc.cli.infer_fmri_manifest "${args[@]}"

FIRST_SUBJECT="${MIRAGE_SUBJECTS%%,*}"
PREDICTION_PATH="${OUTPUT_DIR}/online_sample_${FIRST_SUBJECT}.npy"
GLASS_MEAN_PATH="${OUTPUT_DIR}/online_sample_${FIRST_SUBJECT}_glass_mean.png"
GLASS_VIDEO_PATH="${OUTPUT_DIR}/online_sample_${FIRST_SUBJECT}_glass_brain.mp4"
GLASS_FRAME_DIR="${OUTPUT_DIR}/online_sample_${FIRST_SUBJECT}_glass_frames"

PREDICTION_PATH="${PREDICTION_PATH}" \
GLASS_MEAN_PATH="${GLASS_MEAN_PATH}" \
GLASS_VIDEO_PATH="${GLASS_VIDEO_PATH}" \
GLASS_FRAME_DIR="${GLASS_FRAME_DIR}" \
MIRAGE_VIDEO_FPS="${MIRAGE_VIDEO_FPS:-6}" \
MIRAGE_VIDEO_MAX_FRAMES="${MIRAGE_VIDEO_MAX_FRAMES:-100}" \
XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import imageio.v2 as imageio
import nibabel as nib
import numpy as np
from nilearn import datasets, plotting


def parcels_to_volume(values: np.ndarray):
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=1000,
        yeo_networks=7,
        resolution_mm=2,
        data_dir=os.environ.get("XDG_CACHE_HOME"),
        verbose=0,
    )
    atlas_img = nib.load(atlas.maps)
    labels = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    volume = np.zeros(labels.shape, dtype=np.float32)
    for parcel_idx, value in enumerate(values, start=1):
        volume[labels == parcel_idx] = float(value)
    return nib.Nifti1Image(volume, atlas_img.affine, atlas_img.header)


prediction_path = Path(os.environ["PREDICTION_PATH"])
pred = np.load(prediction_path).astype(np.float32, copy=False)
if pred.ndim != 2 or pred.shape[1] != 1000:
    raise SystemExit(f"Expected prediction shape (n_trs, 1000), got {pred.shape}")

mean_values = pred.mean(axis=0)

mean_img = parcels_to_volume(mean_values)

mean_path = Path(os.environ["GLASS_MEAN_PATH"])
video_path = Path(os.environ["GLASS_VIDEO_PATH"])
frame_dir = Path(os.environ["GLASS_FRAME_DIR"])
mean_path.parent.mkdir(parents=True, exist_ok=True)
frame_dir.mkdir(parents=True, exist_ok=True)

plotting.plot_glass_brain(
    mean_img,
    display_mode="lyrz",
    colorbar=True,
    plot_abs=False,
    title="MIRAGE mean predicted response",
    output_file=str(mean_path),
)

fps = float(os.environ["MIRAGE_VIDEO_FPS"])
max_frames = max(1, int(os.environ["MIRAGE_VIDEO_MAX_FRAMES"]))
frame_indices = np.linspace(
    0,
    pred.shape[0] - 1,
    num=min(max_frames, pred.shape[0]),
    dtype=int,
)
frame_vmax = float(np.nanmax(np.abs(pred)))
if not np.isfinite(frame_vmax) or frame_vmax <= 0.0:
    frame_vmax = 1.0
frame_paths = []
for frame_number, tr_idx in enumerate(frame_indices):
    frame_path = frame_dir / f"frame_{frame_number:04d}.png"
    frame_img = parcels_to_volume(pred[int(tr_idx)])
    plotting.plot_glass_brain(
        frame_img,
        display_mode="lyrz",
        colorbar=True,
        plot_abs=False,
        vmin=-frame_vmax,
        vmax=frame_vmax,
        symmetric_cbar=True,
        title=f"MIRAGE predicted response, TR {int(tr_idx)}",
        output_file=str(frame_path),
    )
    frame_paths.append(frame_path)

with imageio.get_writer(video_path, fps=fps, macro_block_size=16) as writer:
    for frame_path in frame_paths:
        writer.append_data(imageio.imread(frame_path))

print(f"Wrote glass-brain mean map: {mean_path}")
print(f"Wrote glass-brain video: {video_path}")
PY

echo
echo "Demo outputs:"
echo "  predictions: ${OUTPUT_DIR}"
echo "  summary:     ${OUTPUT_DIR}/manifest_inference_summary.csv"
echo "  glass mean:  ${GLASS_MEAN_PATH}"
echo "  glass video: ${GLASS_VIDEO_PATH}"
