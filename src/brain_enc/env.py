"""Environment loading and variable validation for brain_enc."""

import os
from pathlib import Path


def load_env(dotenv_path: Path | None = None) -> None:
    """Load .env from repo root (or a given path) without overriding existing vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if dotenv_path is None:
        # src/brain_enc/env.py -> src/brain_enc/ -> src/ -> repo root
        dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path, override=False)


def repo_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[2]


def _resolve_relative_to_scratch(env_var: str, default_relpath: str) -> Path:
    scratch = get_scratchpath()
    rel_or_abs = os.environ.get(env_var, default_relpath)
    path = Path(rel_or_abs)
    return path if path.is_absolute() else scratch / path


def get_scratchpath() -> Path:
    load_env()
    val = os.environ.get("SCRATCHPATH")
    if not val:
        raise EnvironmentError(
            "SCRATCHPATH is not set. "
            "Export it in your shell or add it to the .env file at the repo root."
        )
    return Path(val)


def get_datapath() -> Path:
    load_env()
    val = os.environ.get("DATAPATH")
    if val:
        return Path(val)
    return _resolve_relative_to_scratch("DATASET_PATH", "datasets/algonauts_2025")


def get_outputpath() -> Path:
    load_env()
    val = os.environ.get("OUTPUTPATH") or os.environ.get("SAVEPATH")
    if val:
        return Path(val)
    return _resolve_relative_to_scratch("OUTPUT_PATH", "outputs/mirage")


def get_local_outputpath() -> Path:
    """Return the repo-local output root for browsable analysis artifacts.

    ``LOCAL_OUTPUT_PATH`` is resolved relative to the repository root rather
    than ``SCRATCHPATH`` so that figures and analysis reports stay beside the
    code while training checkpoints live on the shared output root.
    """

    load_env()
    path = Path(os.environ.get("LOCAL_OUTPUT_PATH", "outputs"))
    return path if path.is_absolute() else repo_root() / path


def get_savepath() -> Path:
    """Alias for the output root."""
    return get_outputpath()


def get_slurm_partition() -> str:
    load_env()
    return os.environ.get("SLURM_PARTITION", "")


def get_wandb_project() -> str:
    load_env()
    return os.environ.get("WANDB_PROJECT", "mirage")


def get_wandb_entity() -> str | None:
    load_env()
    return os.environ.get("WANDB_ENTITY") or None
