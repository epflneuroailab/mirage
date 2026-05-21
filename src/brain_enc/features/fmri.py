"""Sprint-local fMRI parcel extractor.

Reads parcellated BOLD time-series from the Algonauts 2025 HDF5 files,
z-scores each parcel time-series, and writes the result to the fMRI
feature store.

Note on cache dtype: ``extract_fmri()`` returns ``float32`` arrays, but the
extraction CLI may cast them just before HDF5 write via ``--save-dtype``. The
generic feature-store reader preserves the stored dtype by default, and the
training loader now carries cached arrays through with that stored dtype.

Organizer HDF5 layout
---------------------
    file: sub-{XX}_task-{friends|movie10}_..._bold.h5
    key:  ses-{NNN}_task-{stimulus}        (e.g. ses-003_task-s01e01a)
    shape: (n_trs, 1000)  — time × parcels

Logical tensor layout
---------------------
The feature pipeline keeps fMRI time-major end to end:
``(n_trs, 1000)`` (time × parcels) in memory and on disk.
"""


import logging
from pathlib import Path

import h5py
import numpy as np

from brain_enc.data.feature_store import FeatureOutput

logger = logging.getLogger(__name__)


def extract_fmri(
    fmri_h5_path: str | Path,
    fmri_h5_key: str,
    z_score: bool = True,
) -> FeatureOutput:
    """Load one BOLD run from the HDF5 file and return a FeatureOutput.

    Parameters
    ----------
    fmri_h5_path:
        Absolute path to the subject's fMRI HDF5 file.
    fmri_h5_key:
        The dataset key within the HDF5 (e.g. "ses-003_task-s01e01a").
    z_score:
        If True, z-score each parcel time-series independently
        (mean 0, std 1 across the time axis).

    Returns
    -------
    FeatureOutput
        ``features``   — float32 array of shape (n_trs, n_parcels)
        ``time_axis``  — TR indices as float array  (n_trs,)
        ``layer_axis`` — ``None`` (fMRI has no layer dimension)
        ``metadata``   — dict with path / key / n_parcels / n_trs
    """
    fmri_h5_path = Path(fmri_h5_path)
    if not fmri_h5_path.exists():
        raise FileNotFoundError(f"fMRI HDF5 not found: {fmri_h5_path}")

    with h5py.File(fmri_h5_path, "r") as f:
        if fmri_h5_key not in f:
            raise KeyError(
                f"Key '{fmri_h5_key}' not found in {fmri_h5_path}. "
                f"Available: {list(f.keys())[:10]}"
            )
        data = f[fmri_h5_key][:].astype(np.float32)  # (n_trs, n_parcels)

    if z_score:
        mu = data.mean(axis=0, keepdims=True)
        sigma = data.std(axis=0, ddof=1, keepdims=True)  # sample std, matches nilearn zscore_sample
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        data = (data - mu) / sigma

    n_trs, n_parcels = data.shape
    time_axis = np.arange(n_trs, dtype=np.float32)

    return FeatureOutput(
        features=data,
        time_axis=time_axis,
        layer_axis=None,
        metadata={
            "fmri_h5_path": str(fmri_h5_path),
            "fmri_h5_key": fmri_h5_key,
            "n_parcels": n_parcels,
            "n_trs": n_trs,
            "z_scored": z_score,
        },
    )
