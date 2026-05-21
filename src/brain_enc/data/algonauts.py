"""Algonauts 2025 dataset manifest and metadata loader.

Builds a flat DataFrame with one row per (subject, stimulus-run):

    subject            — e.g. "sub-01"
    subject_idx        — 0-based integer index (sorted alphabetically)
    task               — "friends" | "movie10"
    movie              — season str for friends (e.g. "1"), clip name for movie10
    stimulus_id        — stimulus-level unique id, e.g. "friends/s01e01a"
                         or "movie10/life01_run1"  (shared across subjects)
    fmri_item_id       — per-subject key:  "{subject}/{stimulus_id}"
    chunk              — split unit, e.g. "chunk:e01a" or "chunk:1"
    fmri_h5_path       — absolute path to the HDF5 file
    fmri_h5_key        — key within the HDF5 (e.g. "ses-003_task-s01e01a")
    video_path         — absolute path to the .mkv file
    transcript_path    — absolute path to the .tsv transcript
    n_parcels          — number of parcels (1000)
    split              — "train" | "val" | "test"  (filled by add_splits)

Dataset layout (datalad download root = DATAPATH):

    fmri/
        sub-{XX}/func/
            sub-{XX}_task-friends_..._desc-s123456_bold.h5
            sub-{XX}_task-movie10_..._bold.h5
    stimuli/
        movies/
            friends/s{N}/friends_s{NN}{chunk}.mkv
            movie10/{movie}/{movie}{NN}.mkv
        transcripts/
            friends/s{N}/friends_s{NN}{chunk}.tsv
            movie10/{movie}/movie10_{movie}{NN}.tsv
            ood/{movie}/ood_{movie}{split}.tsv

fMRI HDF5 key format:
    friends  → ses-{NNN}_task-s{season:02d}{chunk}  (e.g. ses-003_task-s01e01a)
    movie10  → ses-{NNN}_task-{movie}{chunk:02d}[_run-{run}]
"""

from __future__ import annotations

import ast
import logging
import re
import typing as tp
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from brain_enc._moviepy import VideoFileClip

logger = logging.getLogger(__name__)

N_PARCELS = 1000
FMRI_TR = 1.49       # seconds
HRF_DELAY = 4.47     # seconds
WINDOW_N_TRS = 100
WINDOW_DURATION_S = WINDOW_N_TRS * FMRI_TR
WINDOW_STRIDE_S = WINDOW_DURATION_S
FEATURE_HZ = 2.0
WINDOW_N_FEATURE_FRAMES = int(round(WINDOW_DURATION_S * FEATURE_HZ))

# Subjects present in the challenge
_SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]

# Friends: seasons 1-6 are training; season 7 is test (stimuli only, no fMRI)
_FRIENDS_TRAIN_SEASONS = list(range(1, 7))
_FRIENDS_TEST_SEASON = 7

# Episodes per season (Friends has variable episode counts)
_FRIENDS_EPISODES_PER_SEASON = {
    1: 24, 2: 24, 3: 25, 4: 24, 5: 24, 6: 25, 7: 24,
}
_FRIENDS_PARTS = list("abcd")

# Movie10 movies and their number of chunks
_MOVIE10_CHUNKS = {
    "bourne": 10,
    "wolf": 17,
    "life": 5,
    "figures": 12,
}
# life and figures were watched twice (run-1, run-2); bourne/wolf only once
_MOVIE10_RUNS: dict[str, list[int]] = {
    "bourne": [0],
    "wolf": [0],
    "life": [1, 2],
    "figures": [1, 2],
}

_OOD_MOVIES = [
    "chaplin",
    "mononoke",
    "passepartout",
    "planetearth",
    "pulpfiction",
    "wot",
]
_OOD_SPLITS = [1, 2]

# Some friends stimuli are absent in the dataset — skip if transcript missing
_FRIENDS_SKIP: set[tuple[int, int, str]] = {
    (5, 20, "a"), (4, 1, "a"), (6, 3, "a"), (4, 13, "b"), (4, 1, "b"),
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _friends_video_path(root: Path, season: int, episode: int, part: str) -> Path:
    return (
        root / "stimuli" / "movies" / "friends"
        / f"s{season}"
        / f"friends_s{season:02d}e{episode:02d}{part}.mkv"
    )


def _friends_transcript_path(root: Path, season: int, episode: int, part: str) -> Path:
    return (
        root / "stimuli" / "transcripts" / "friends"
        / f"s{season}"
        / f"friends_s{season:02d}e{episode:02d}{part}.tsv"
    )


def _movie10_video_path(root: Path, movie: str, chunk: int) -> Path:
    return (
        root / "stimuli" / "movies" / "movie10"
        / movie
        / f"{movie}{chunk:02d}.mkv"
    )


def _movie10_transcript_path(root: Path, movie: str, chunk: int) -> Path:
    return (
        root / "stimuli" / "transcripts" / "movie10"
        / movie
        / f"movie10_{movie}{chunk:02d}.tsv"
    )


def _ood_video_path(root: Path, movie: str, split: int) -> Path:
    base = root / "stimuli" / "movies" / "ood" / movie
    mkv = base / f"task-{movie}{split}_video.mkv"
    if mkv.exists():
        return mkv
    mp4 = base / f"task-{movie}{split}_video.mp4"
    return mp4


def _ood_transcript_path(root: Path, movie: str, split: int) -> Path:
    return (
        root / "stimuli" / "transcripts" / "ood"
        / movie
        / f"ood_{movie}{split}.tsv"
    )


def _fmri_h5_path(root: Path, subject: str, task: str) -> Path:
    stem = (
        f"{subject}_task-{task}_space-MNI152NLin2009cAsym"
        f"_atlas-Schaefer18_parcel-1000Par7Net"
    )
    suffix = "_desc-s123456_bold.h5" if task == "friends" else "_bold.h5"
    return root / "fmri" / subject / "func" / (stem + suffix)


def _find_fmri_key(h5_keys: list[str], match_str: str) -> str | None:
    """Return the single HDF5 key that contains match_str."""
    found = [k for k in h5_keys if match_str in k]
    if len(found) == 1:
        return found[0]
    return None


def _friends_chunk_id(episode: int, part: str) -> str:
    """Return the split id for a Friends stimulus chunk."""
    return f"chunk:e{episode:02d}{part}"


def _movie10_chunk_id(chunk: int) -> str:
    """Return the split id for a Movie10 stimulus chunk."""
    return f"chunk:{chunk}"


# ---------------------------------------------------------------------------
# Subject index
# ---------------------------------------------------------------------------

def _subject_idx(subjects: list[str]) -> dict[str, int]:
    return {s: i for i, s in enumerate(sorted(subjects))}


@lru_cache(maxsize=None)
def _transcript_duration_s(path_str: str) -> float:
    """Return a robust transcript-backed stimulus duration in seconds."""
    if not path_str:
        return 0.0
    path = Path(path_str)
    if not path.exists() or path.is_dir():
        return 0.0

    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    if "words_per_tr" not in df.columns:
        return 0.0

    total_duration = len(df) * FMRI_TR
    for _, row in df.iterrows():
        raw_onsets = row.get("onsets_per_tr", "[]")
        raw_durations = row.get("durations_per_tr", "[]")
        try:
            onsets = ast.literal_eval(raw_onsets)
            durations = ast.literal_eval(raw_durations)
        except (ValueError, SyntaxError):
            continue
        for onset, duration in zip(onsets, durations):
            total_duration = max(
                total_duration,
                float(onset) + max(float(duration), 0.0),
            )
    return float(total_duration)


@lru_cache(maxsize=None)
def _fmri_duration_s(path_str: str, key: str) -> float:
    """Return one run's BOLD duration without materialising the dataset."""
    if not path_str or not key:
        return 0.0
    import h5py

    with h5py.File(path_str, "r") as f:
        if key not in f:
            return 0.0
        n_trs = int(f[key].shape[0])
    return float(n_trs * FMRI_TR)


@lru_cache(maxsize=None)
def _audio_duration_s(path_str: str) -> float:
    """Return audio duration in seconds without loading the full waveform."""
    path = Path(path_str)
    if not path.exists():
        return 0.0

    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    return 0.0


@lru_cache(maxsize=None)
def _video_duration_s(path_str: str) -> float:
    """Return video duration in seconds without decoding all frames."""
    path = Path(path_str)
    if not path.exists():
        return 0.0

    # Preferred path: moviepy reports container duration robustly.
    try:
        clip = VideoFileClip(str(path), audio=False)
        duration = float(getattr(clip, "duration", 0.0) or 0.0)
        clip.close()
        if duration > 0.0:
            return duration
    except Exception:
        pass

    # Fallback path: OpenCV metadata.
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        if fps > 0.0 and n_frames > 0.0:
            return n_frames / fps
    except Exception:
        pass

    return 0.0


def _prepare_strided_windows(
    start: float,
    stop: float,
    stride: float,
    duration: float,
    drop_incomplete: bool = False,
) -> tuple[list[float], list[float]]:
    """Return fixed-duration window starts for segmenting stimulus runs."""
    if duration <= 0.0:
        raise ValueError(f"duration must be > 0, got {duration}")
    if stride <= 0.0:
        raise ValueError(f"stride must be > 0, got {stride}")
    if stop <= start:
        return [start], [duration]

    if drop_incomplete:
        stop -= duration

    eps = 1e-8
    # Match the reference event-grid behavior: np.arange(start, stop + eps, stride)
    starts = np.arange(start, stop + eps, stride).astype(float).tolist()
    if not starts:
        starts = [float(start)]
    durations = [float(duration)] * len(starts)
    return starts, durations


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(datapath: str | Path | None = None) -> pd.DataFrame:
    """Scan the Algonauts 2025 download tree and return a manifest DataFrame.

    Parameters
    ----------
    datapath:
        Root of the datalad download (the directory containing ``fmri/`` and
        ``stimuli/``).  If ``None``, reads ``DATAPATH`` from the environment.
    """
    if datapath is None:
        from brain_enc.env import get_datapath
        datapath = get_datapath()
    datapath = Path(datapath)
    if not datapath.exists():
        raise FileNotFoundError(f"Dataset root not found: {datapath}")

    from brain_enc.features.base import progress_iter

    rows: list[dict] = []

    subject_iter = progress_iter(
        _SUBJECTS,
        desc="Manifest subjects",
        total=len(_SUBJECTS),
        leave=False,
        unit="subject",
    )
    for subject in subject_iter:
        friends_h5 = _fmri_h5_path(datapath, subject, "friends")
        movie10_h5 = _fmri_h5_path(datapath, subject, "movie10")

        # Load HDF5 key lists once per subject (avoid reopening per stimulus)
        friends_keys: list[str] = []
        movie10_keys: list[str] = []
        if friends_h5.exists():
            import h5py
            with h5py.File(friends_h5, "r") as f:
                friends_keys = list(f.keys())
        if movie10_h5.exists():
            import h5py
            with h5py.File(movie10_h5, "r") as f:
                movie10_keys = list(f.keys())

        if not friends_keys and not movie10_keys:
            logger.warning("No HDF5 files found for subject %s, skipping", subject)
            continue

        # --- Friends ---
        for season in _FRIENDS_TRAIN_SEASONS:
            for episode in range(1, _FRIENDS_EPISODES_PER_SEASON[season] + 1):
                for part in _FRIENDS_PARTS:
                    if (season, episode, part) in _FRIENDS_SKIP:
                        continue
                    transcript = _friends_transcript_path(datapath, season, episode, part)
                    if not transcript.exists():
                        continue  # stimulus absent in this download
                    video = _friends_video_path(datapath, season, episode, part)
                    chunk_str = f"e{episode:02d}{part}"
                    key_match = f"s{season:02d}{chunk_str}"  # e.g. s01e01a
                    fmri_key = _find_fmri_key(friends_keys, key_match)
                    stimulus_id = f"friends/s{season:02d}e{episode:02d}{part}"
                    rows.append(dict(
                        subject=subject,
                        task="friends",
                        movie=str(season),
                        stimulus_id=stimulus_id,
                        chunk=_friends_chunk_id(episode, part),
                        fmri_h5_path=str(friends_h5),
                        fmri_h5_key=fmri_key or "",
                        video_path=str(video),
                        transcript_path=str(transcript),
                        n_parcels=N_PARCELS,
                    ))

        # --- Movie10 ---
        for movie, n_chunks in _MOVIE10_CHUNKS.items():
            runs = _MOVIE10_RUNS[movie]
            for chunk in range(1, n_chunks + 1):
                transcript = _movie10_transcript_path(datapath, movie, chunk)
                if not transcript.exists():
                    continue
                video = _movie10_video_path(datapath, movie, chunk)
                for run in runs:
                    run_suffix = f"_run-{run}" if run > 0 else ""
                    key_match = f"{movie}{chunk:02d}{run_suffix}"
                    fmri_key = _find_fmri_key(movie10_keys, key_match)
                    run_tag = f"_run{run}" if run > 0 else ""
                    stimulus_id = f"movie10/{movie}{chunk:02d}{run_tag}"
                    rows.append(dict(
                        subject=subject,
                        task="movie10",
                        movie=movie,
                        stimulus_id=stimulus_id,
                        chunk=_movie10_chunk_id(chunk),
                        fmri_h5_path=str(movie10_h5),
                        fmri_h5_key=fmri_key or "",
                        video_path=str(video),
                        transcript_path=str(transcript),
                        n_parcels=N_PARCELS,
                    ))

    if not rows:
        raise RuntimeError(
            f"No stimuli found under {datapath}. "
            "Ensure the dataset has been downloaded and DATAPATH is correct."
        )

    df = pd.DataFrame(rows)

    # Subject index
    idx_map = _subject_idx(df["subject"].unique().tolist())
    df["subject_idx"] = df["subject"].map(idx_map)

    # Per-subject fMRI item key (used by fMRI feature store)
    df["fmri_item_id"] = df["subject"] + "/" + df["stimulus_id"]

    df["split"] = ""

    missing_fmri = int((df["fmri_h5_key"] == "").sum())
    if missing_fmri:
        logger.warning(
            "Manifest contains %d rows without fmri_h5_key. "
            "Training and fMRI extraction should validate the manifest before use.",
            missing_fmri,
        )

    logger.info(
        "Manifest: %d rows | %d subjects | tasks=%s",
        len(df),
        df["subject"].nunique(),
        df["task"].unique().tolist(),
    )
    return df


# ---------------------------------------------------------------------------
# Friends Season 7 — stimuli-only manifest (no fMRI labels; used for submission)
# ---------------------------------------------------------------------------

def build_s7_manifest(datapath: str | Path | None = None) -> pd.DataFrame:
    """Return a DataFrame of Friends Season 7 stimuli (no fMRI, no subject dim).

    Each row is one stimulus chunk.  ``fmri_h5_path`` and ``fmri_h5_key`` are
    empty strings because S7 has no paired fMRI in the challenge data.

    This manifest is used by the evaluation and submission pipelines to generate
    predictions for the held-out test set.
    """
    if datapath is None:
        from brain_enc.env import get_datapath
        datapath = get_datapath()
    datapath = Path(datapath)
    from brain_enc.features.base import progress_iter

    rows: list[dict] = []
    season = _FRIENDS_TEST_SEASON
    episode_iter = progress_iter(
        range(1, _FRIENDS_EPISODES_PER_SEASON[season] + 1),
        desc="Manifest Friends S7",
        total=_FRIENDS_EPISODES_PER_SEASON[season],
        leave=False,
        unit="episode",
    )
    for episode in episode_iter:
        for part in _FRIENDS_PARTS:
            if (season, episode, part) in _FRIENDS_SKIP:
                continue
            transcript = _friends_transcript_path(datapath, season, episode, part)
            if not transcript.exists():
                continue
            video = _friends_video_path(datapath, season, episode, part)
            stimulus_id = f"friends/s{season:02d}e{episode:02d}{part}"
            rows.append(dict(
                task="friends",
                movie=str(season),
                stimulus_id=stimulus_id,
                chunk=_friends_chunk_id(episode, part),
                subject="",
                subject_idx=-1,
                fmri_item_id="",
                fmri_h5_path="",
                fmri_h5_key="",
                video_path=str(video),
                transcript_path=str(transcript),
                n_parcels=N_PARCELS,
                split="test",
            ))

    if not rows:
        logger.warning(
            "No Friends S7 stimuli found under %s — "
            "check that season 7 videos/transcripts have been downloaded.",
            datapath,
        )

    df = pd.DataFrame(rows)
    logger.info("Friends S7 manifest: %d stimuli", len(df))
    return df


def build_ood_manifest(datapath: str | Path | None = None) -> pd.DataFrame:
    """Return a DataFrame of OOD movie stimuli (no fMRI, no subject dim).

    Each row is one OOD movie split. ``fmri_h5_path`` and ``fmri_h5_key`` are
    empty strings because these held-out challenge stimuli do not have paired
    fMRI in the public dataset.

    The stable ``stimulus_id`` is ``ood/<movie><split>``, so future submission
    paths can recover challenge keys such as ``chaplin1`` or ``wot2`` by taking
    the suffix after ``ood/``.
    """
    if datapath is None:
        from brain_enc.env import get_datapath
        datapath = get_datapath()
    datapath = Path(datapath)
    from brain_enc.features.base import progress_iter

    rows: list[dict] = []
    movie_iter = progress_iter(
        _OOD_MOVIES,
        desc="Manifest OOD",
        total=len(_OOD_MOVIES),
        leave=False,
        unit="movie",
    )
    for movie in movie_iter:
        for split in _OOD_SPLITS:
            video = _ood_video_path(datapath, movie, split)
            if not video.exists():
                continue
            transcript = _ood_transcript_path(datapath, movie, split)
            rows.append(dict(
                task="ood",
                movie=movie,
                stimulus_id=f"ood/{movie}{split}",
                chunk=f"chunk:{movie}{split}",
                subject="",
                subject_idx=-1,
                fmri_item_id="",
                fmri_h5_path="",
                fmri_h5_key="",
                video_path=str(video),
                transcript_path=str(transcript if transcript.exists() else ""),
                n_parcels=N_PARCELS,
                split="test",
            ))

    if not rows:
        logger.warning(
            "No OOD stimuli found under %s — "
            "check that OOD videos have been downloaded.",
            datapath,
        )

    df = pd.DataFrame(rows)
    logger.info("OOD manifest: %d stimuli", len(df))
    return df


def build_stimulus_manifest(
    datapath: str | Path | None,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Append stimulus-only Algonauts splits used for extraction/evaluation."""
    extra_manifests = []

    s7_manifest = build_s7_manifest(datapath)
    if not s7_manifest.empty:
        extra_manifests.append(("Friends S7", s7_manifest))

    ood_manifest = build_ood_manifest(datapath)
    if not ood_manifest.empty:
        extra_manifests.append(("OOD", ood_manifest))

    if not extra_manifests:
        return manifest

    combined = pd.concat(
        [manifest] + [df for _, df in extra_manifests],
        ignore_index=True,
        sort=False,
    )
    for label, df in extra_manifests:
        logger.info(
            "Appended %s stimuli for feature extraction: +%d rows",
            label,
            len(df),
        )
    logger.info("Stimulus extraction manifest now has %d rows total", len(combined))
    return combined


def annotate_manifest_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with stable duration columns populated."""
    from brain_enc.features.base import progress_iter

    out = df.copy()
    transcript_values = out.get("transcript_path", pd.Series([""] * len(out), index=out.index))
    video_values = out.get("video_path", pd.Series([""] * len(out), index=out.index))
    audio_values = out.get("audio_path", pd.Series([""] * len(out), index=out.index))
    fmri_paths = out.get("fmri_h5_path", pd.Series([""] * len(out), index=out.index))
    fmri_keys = out.get("fmri_h5_key", pd.Series([""] * len(out), index=out.index))

    out["transcript_duration_s"] = [
        _transcript_duration_s(str(value))
        for value in progress_iter(
            transcript_values,
            desc="Manifest durations: transcript",
            total=len(out),
            leave=False,
            unit="row",
        )
    ]
    out["video_duration_s"] = [
        _video_duration_s(str(value))
        for value in progress_iter(
            video_values,
            desc="Manifest durations: video",
            total=len(out),
            leave=False,
            unit="row",
        )
    ]
    out["audio_duration_s"] = [
        _audio_duration_s(str(value))
        for value in progress_iter(
            audio_values,
            desc="Manifest durations: audio",
            total=len(out),
            leave=False,
            unit="row",
        )
    ]
    out["fmri_duration_s"] = [
        _fmri_duration_s(str(path), str(key))
        for path, key in progress_iter(
            zip(fmri_paths, fmri_keys),
            desc="Manifest durations: fmri",
            total=len(out),
            leave=False,
            unit="row",
        )
    ]
    return out


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def add_splits(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    seed: int = 33,
    *,
    split_strategy: str = "chunk",
    holdout_friends_season: int | None = None,
    custom_val_set: str | list[str] | tuple[str, ...] | None = None,
    custom_val_name: str | None = None,
) -> pd.DataFrame:
    """Assign 'train' / 'val' splits (no held-out test within this manifest).

    Friends season 7 and OOD challenge stimuli are NOT in this manifest (they
    have no fMRI labels).  All rows here are eligible for train/val.

    Supported strategies:

    - ``chunk``: deterministic hashing over event-table
      chunk ids like ``"chunk:e01a"`` or ``"chunk:1"``
    - ``friends_season_holdout``: mark one full Friends season as validation,
      keeping all other Friends seasons and Movie10 in train; this mirrors the
      public challenge tutorial's default ``movies_train``/``movies_val``
      pattern
    - ``custom_holdout``: mark rows matching a comma-separated selector list as
      validation, e.g. ``"s6-18b,figures"``. Supported selectors are Friends
      seasons/chunks (``s6``, ``s6-18b``, ``s06e18b``) and Movie10
      movies/chunks (``figures``, ``figures03``).
    """
    from brain_enc.data.splits import ChunkSplitter

    df = df.copy()
    df["split"] = "train"

    if split_strategy == "chunk":
        chunks = df["chunk"].unique().tolist()
        splitter = ChunkSplitter(val_ratio=val_ratio, seed=seed)
        chunk_split = splitter.split(chunks)
        df["split"] = df["chunk"].map(chunk_split)
    elif split_strategy == "friends_season_holdout":
        season = 6 if holdout_friends_season is None else int(holdout_friends_season)
        if season not in _FRIENDS_TRAIN_SEASONS:
            raise ValueError(
                "holdout_friends_season must be one of the Friends training seasons "
                f"{_FRIENDS_TRAIN_SEASONS}, got {season}"
            )
        season_str = str(season)
        is_holdout = (df["task"] == "friends") & (df["movie"].astype(str) == season_str)
        if not bool(is_holdout.any()):
            raise ValueError(
                "friends_season_holdout selected, but manifest has no Friends rows for "
                f"season {season}"
            )
        df.loc[is_holdout, "split"] = "val"
    elif split_strategy == "custom_holdout":
        selectors = _normalize_custom_val_selectors(custom_val_set)
        is_holdout = _custom_holdout_mask(df, selectors)
        if not bool(is_holdout.any()):
            raise ValueError(
                "custom_holdout selected, but no manifest rows matched custom_val_set="
                f"{selectors!r}"
            )
        df.loc[is_holdout, "split"] = "val"
    else:
        raise ValueError(
            "split_strategy must be 'chunk', 'friends_season_holdout', or "
            "'custom_holdout', got "
            f"{split_strategy!r}"
        )

    counts = df["split"].value_counts().to_dict()
    unique_counts: dict[str, int] | None = None
    if "stimulus_id" in df.columns:
        unique_counts = df.groupby("split")["stimulus_id"].nunique().to_dict()
    logger.info(
        "Split counts (%s): rows=%s unique_stimuli=%s",
        custom_val_name or split_strategy,
        counts,
        unique_counts,
    )
    return df


def _normalize_custom_val_selectors(
    custom_val_set: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Return non-empty lowercase custom holdout selectors."""
    if custom_val_set is None:
        raise ValueError("custom_holdout requires data.custom_val_set or --val-set")
    if isinstance(custom_val_set, str):
        raw_parts = custom_val_set.split(",")
    else:
        raw_parts = []
        for value in custom_val_set:
            raw_parts.extend(str(value).split(","))
    selectors = [part.strip().lower() for part in raw_parts if part.strip()]
    if not selectors:
        raise ValueError("custom_holdout requires at least one non-empty validation selector")
    return selectors


def _custom_holdout_mask(df: pd.DataFrame, selectors: list[str]) -> pd.Series:
    """Return a boolean mask for rows selected by custom validation selectors."""
    mask = pd.Series(False, index=df.index)
    for selector in selectors:
        selector_mask = _custom_selector_mask(df, selector)
        if not bool(selector_mask.any()):
            raise ValueError(
                f"custom validation selector {selector!r} matched no manifest rows"
            )
        mask = mask | selector_mask
    return mask


def _custom_selector_mask(df: pd.DataFrame, selector: str) -> pd.Series:
    """Return rows matching a single Friends or Movie10 selector."""
    normalized = selector.strip().lower()
    if normalized.startswith("friends/"):
        normalized = normalized.removeprefix("friends/")
    if normalized.startswith("movie10/"):
        normalized = normalized.removeprefix("movie10/")
    normalized = normalized.replace("_", "-")

    friends_match = re.fullmatch(
        r"s0?(?P<season>[1-7])(?:-?e?(?P<episode>\d{1,2})(?P<part>[a-d]))?",
        normalized,
    )
    if friends_match is not None:
        season = int(friends_match.group("season"))
        episode = friends_match.group("episode")
        part = friends_match.group("part")
        if episode is None:
            return (df["task"] == "friends") & (df["movie"].astype(str) == str(season))
        stimulus_id = f"friends/s{season:02d}e{int(episode):02d}{part}"
        return df["stimulus_id"].astype(str) == stimulus_id

    movie_names = "|".join(sorted(_MOVIE10_CHUNKS))
    movie_match = re.fullmatch(
        rf"(?P<movie>{movie_names})(?:-?(?P<chunk>\d{{1,2}}))?",
        normalized.replace("/", "-"),
    )
    if movie_match is not None:
        movie = movie_match.group("movie")
        chunk = movie_match.group("chunk")
        mask = (df["task"] == "movie10") & (df["movie"].astype(str) == movie)
        if chunk is not None:
            mask = mask & (df["chunk"].astype(str) == f"chunk:{int(chunk)}")
        return mask

    raise ValueError(
        f"Unknown custom validation selector {selector!r}. Expected Friends selectors "
        "like 's6' or 's6-18b', or Movie10 selectors like 'figures' or 'figures03'."
    )


def build_segment_manifest(
    df: pd.DataFrame,
    *,
    window_duration_s: float = WINDOW_DURATION_S,
    stride_s: float = WINDOW_STRIDE_S,
    hrf_delay_s: float = HRF_DELAY,
) -> pd.DataFrame:
    """Expand a per-run manifest into fixed-duration training windows.

    Each returned row corresponds to one fixed-duration window over a single
    stimulus run, with segment timing stored in the shifted stimulus frame:
    ``segment_start_s = -hrf_delay_s + k * stride_s``.
    """
    rows: list[dict[str, tp.Any]] = []
    for row in df.to_dict(orient="records"):
        raw_transcript_duration = row.get("transcript_duration_s")
        raw_fmri_duration = row.get("fmri_duration_s")
        raw_video_duration = row.get("video_duration_s")
        raw_audio_duration = row.get("audio_duration_s")
        transcript_duration_s = float(
            raw_transcript_duration
            if raw_transcript_duration is not None and not pd.isna(raw_transcript_duration)
            else _transcript_duration_s(str(row.get("transcript_path", "")))
        )
        fmri_duration_s = float(
            raw_fmri_duration
            if raw_fmri_duration is not None and not pd.isna(raw_fmri_duration)
            else _fmri_duration_s(
                str(row.get("fmri_h5_path", "")),
                str(row.get("fmri_h5_key", "")),
            )
        )
        video_duration_s = float(
            raw_video_duration
            if raw_video_duration is not None and not pd.isna(raw_video_duration)
            else _video_duration_s(str(row.get("video_path", "")))
        )
        audio_duration_s = float(
            raw_audio_duration
            if raw_audio_duration is not None and not pd.isna(raw_audio_duration)
            else _audio_duration_s(str(row.get("audio_path", "")))
        )
        stop_s = max(
            transcript_duration_s,
            fmri_duration_s,
            video_duration_s,
            audio_duration_s,
        )
        starts, durations = _prepare_strided_windows(
            start=-hrf_delay_s,
            stop=stop_s - hrf_delay_s,
            stride=stride_s,
            duration=window_duration_s,
            drop_incomplete=False,
        )
        segment_prefix = str(row.get("fmri_item_id", "") or "").strip()
        if not segment_prefix:
            segment_prefix = str(row["stimulus_id"])

        for segment_idx, (segment_start_s, segment_duration_s) in enumerate(zip(starts, durations)):
            seg_row = dict(row)
            seg_row["segment_idx"] = int(segment_idx)
            seg_row["segment_id"] = f"{segment_prefix}@seg{segment_idx:03d}"
            seg_row["segment_start_s"] = float(segment_start_s)
            seg_row["segment_duration_s"] = float(segment_duration_s)
            seg_row["segment_n_trs"] = WINDOW_N_TRS
            seg_row["segment_n_feature_frames"] = WINDOW_N_FEATURE_FRAMES
            rows.append(seg_row)

    out = pd.DataFrame(rows)
    logger.info(
        "Segment manifest: %d rows expanded from %d stimulus runs",
        len(out),
        len(df),
    )
    return out


def ensure_complete_fmri_manifest(
    df: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    """Validate that every manifest row has a resolved fMRI HDF5 key.

    This is intended for code paths that require paired fMRI labels, such as
    training and fMRI feature extraction. Feature-only stimulus extraction can
    still operate on manifests without paired fMRI rows.
    """
    missing = df[df["fmri_h5_key"] == ""]
    if missing.empty:
        return df

    examples = ", ".join(
        f"{row.subject}:{row.stimulus_id}"
        for row in missing[["subject", "stimulus_id"]].head(5).itertuples(index=False)
    )
    raise ValueError(
        f"Manifest validation failed for {context}: "
        f"{len(missing)} rows are missing fmri_h5_key. "
        "This usually means DATAPATH points to an incomplete or mismatched "
        f"Algonauts download. Examples: {examples}"
    )
