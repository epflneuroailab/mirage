"""Helpers for loading trusted checkpoints."""


from pathlib import Path
from typing import Any

import torch


def load_trusted_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    """Load a trusted checkpoint with full pickle support enabled."""
    return torch.load(
        Path(checkpoint_path),
        map_location=map_location,
        weights_only=False,
    )


def load_lightning_module_state(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a full LightningModule state dict into *module*."""
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain a 'state_dict'.")
    module.load_state_dict(state_dict, strict=strict)
    return checkpoint


def load_prefixed_submodule_state(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    prefix: str,
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a prefixed submodule state dict from a trusted Lightning checkpoint."""
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain a 'state_dict'.")

    submodule_state = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not submodule_state:
        raise KeyError(
            f"Checkpoint {checkpoint_path} does not contain parameters with prefix {prefix!r}."
        )
    module.load_state_dict(submodule_state, strict=strict)
    return checkpoint


def load_model_state(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    lightning_prefix: str = "model.",
    map_location: str | torch.device | None = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load either a public safetensors state dict or a trusted Lightning checkpoint."""

    path = Path(checkpoint_path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(path), device=str(map_location or "cpu"))
        module.load_state_dict(state_dict, strict=strict)
        return {"state_dict": state_dict, "format": "safetensors"}

    checkpoint = load_trusted_checkpoint(path, map_location=map_location)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain a 'state_dict'.")

    if any(key.startswith(lightning_prefix) for key in state_dict):
        state_dict = {
            key.removeprefix(lightning_prefix): value
            for key, value in state_dict.items()
            if key.startswith(lightning_prefix)
        }
    module.load_state_dict(state_dict, strict=strict)
    return checkpoint
