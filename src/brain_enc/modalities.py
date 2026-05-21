"""Shared modality helpers for conditioned multimodal extraction."""


import typing as tp

Modality = tp.Literal["text", "audio", "vision"]

MODALITIES: tuple[Modality, ...] = ("text", "audio", "vision")
_MODALITY_ORDER = {name: idx for idx, name in enumerate(MODALITIES)}
_CONDITIONING_ORDER = {"audio": 0, "text": 1, "vision": 2}


def normalize_available_modalities(
    modalities: tp.Iterable[str] | None,
    *,
    target_modality: str | None = None,
) -> tuple[Modality, ...]:
    """Return a canonical, de-duplicated modality tuple.

    The canonical order is ``("audio", "text", "vision")``. When
    ``target_modality`` is given, the returned tuple must contain it.
    """
    if modalities is None:
        if target_modality is None:
            raise ValueError("modalities=None requires target_modality to be set")
        modalities = [target_modality]

    seen: list[Modality] = []
    for modality in modalities:
        if modality not in _MODALITY_ORDER:
            raise ValueError(
                f"Unknown modality {modality!r}. Expected one of {MODALITIES}."
            )
        typed = tp.cast(Modality, modality)
        if typed not in seen:
            seen.append(typed)

    if not seen:
        raise ValueError("available_modalities must be non-empty")
    if (
        target_modality is not None
        and target_modality not in seen
        and not (
            target_modality == "text"
            and seen
            and set(seen).issubset({"audio", "vision"})
        )
    ):
        raise ValueError(
            "available_modalities must include the target modality "
            f"{target_modality!r}; got {seen!r}"
        )

    return tuple(sorted(seen, key=_CONDITIONING_ORDER.__getitem__))


def conditioning_id(
    modalities: tp.Iterable[str] | None,
    *,
    target_modality: str | None = None,
) -> str:
    """Return the canonical conditioning identifier, e.g. ``ctx-audio-text``."""
    normalized = normalize_available_modalities(
        modalities,
        target_modality=target_modality,
    )
    return "ctx-" + "-".join(normalized)
