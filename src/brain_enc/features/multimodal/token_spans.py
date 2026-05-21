"""Stateless token-span helpers for Qwen multimodal extraction."""

from __future__ import annotations

import typing as tp

import numpy as np


def tokenize_with_offsets(
    tokenizer: tp.Any,
    text: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return list(encoded["input_ids"]), [tuple(x) for x in encoded["offset_mapping"]]
    except Exception:
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = list(encoded["input_ids"])
        if not token_ids:
            return [], []
        approx_edges = np.linspace(0, len(text), num=len(token_ids) + 1, dtype=int)
        offsets = [
            (int(approx_edges[i]), int(approx_edges[i + 1]))
            for i in range(len(token_ids))
        ]
        return token_ids, offsets


def find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    stop = len(haystack) - len(needle) + 1
    for start in range(stop):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def locate_transcript_token_span(
    *,
    tokenizer: tp.Any,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    transcript_text: str,
    modality_token_ids: set[int],
) -> tuple[np.ndarray, list[tuple[int, int]], str]:
    if not transcript_text:
        return np.array([], dtype=np.int64), [], "empty_transcript"

    valid_positions = np.flatnonzero(attention_mask.astype(bool))
    valid_ids = input_ids[valid_positions].tolist()

    for prefix in ("", "\n", " ", "\n\n"):
        token_ids, offsets = tokenize_with_offsets(tokenizer, prefix + transcript_text)
        if not token_ids:
            continue
        start = find_subsequence(valid_ids, token_ids)
        if start is None:
            continue

        prefix_len = len(prefix)
        kept_indices: list[int] = []
        kept_offsets: list[tuple[int, int]] = []
        for idx, (char_start, char_stop) in enumerate(offsets):
            if char_stop <= prefix_len or char_start >= prefix_len + len(transcript_text):
                continue
            kept_indices.append(idx)
            kept_offsets.append(
                (
                    max(0, int(char_start) - prefix_len),
                    max(0, int(char_stop) - prefix_len),
                )
            )

        actual = valid_positions[start + np.asarray(kept_indices, dtype=np.int64)]
        strategy = f"subsequence_match:{prefix.encode('unicode_escape').decode()}"
        return actual.astype(np.int64), kept_offsets, strategy

    non_special_positions = [
        pos
        for pos in valid_positions.tolist()
        if int(input_ids[pos]) not in modality_token_ids
    ]
    if not non_special_positions:
        return np.array([], dtype=np.int64), [], "fallback_empty"

    last_mm = max(
        [idx for idx, pos in enumerate(valid_positions.tolist()) if int(input_ids[pos]) in modality_token_ids],
        default=-1,
    )
    if last_mm >= 0:
        non_special_positions = valid_positions[last_mm + 1 :].tolist()
        non_special_positions = [
            pos for pos in non_special_positions if int(input_ids[pos]) not in modality_token_ids
        ]
    if not non_special_positions:
        return np.array([], dtype=np.int64), [], "fallback_empty"

    approx_edges = np.linspace(0, len(transcript_text), num=len(non_special_positions) + 1, dtype=int)
    approx_offsets = [
        (int(approx_edges[i]), int(approx_edges[i + 1]))
        for i in range(len(non_special_positions))
    ]
    return (
        np.asarray(non_special_positions, dtype=np.int64),
        approx_offsets,
        "fallback_non_special_tokens",
    )


def map_token_offsets_to_words(
    *,
    token_offsets: list[tuple[int, int]],
    word_char_spans: list[tuple[int, int]],
) -> list[list[int]]:
    mapping: list[list[int]] = [[] for _ in word_char_spans]
    for token_idx, (tok_start, tok_stop) in enumerate(token_offsets):
        if tok_stop <= tok_start:
            continue
        for word_idx, (word_start, word_stop) in enumerate(word_char_spans):
            if tok_stop <= word_start or tok_start >= word_stop:
                continue
            mapping[word_idx].append(token_idx)
    return mapping


def recover_target_word_token_ids(
    *,
    tokenizer: tp.Any,
    token_offsets: list[tuple[int, int]],
    n_text_tokens: int,
    prefix_text: str,
    previous_prefix_text: str,
    target_word: str,
    target_char_span: tuple[int, int],
) -> tuple[list[int], str]:
    if n_text_tokens <= 0:
        return [], "no_text_positions"

    overlap_ids = [
        token_idx
        for token_idx, (tok_start, tok_stop) in enumerate(token_offsets)
        if tok_stop > target_char_span[0] and tok_start < target_char_span[1]
    ]
    if overlap_ids:
        return overlap_ids, "offset_overlap_scan"

    current_ids, _ = tokenize_with_offsets(tokenizer, prefix_text)
    previous_ids, _ = tokenize_with_offsets(tokenizer, previous_prefix_text)
    delta = len(current_ids) - len(previous_ids)
    if delta > 0:
        delta = min(delta, n_text_tokens)
        return list(range(n_text_tokens - delta, n_text_tokens)), f"suffix_delta:{delta}"

    for variant in (f" {target_word}", target_word, f"\n{target_word}"):
        target_ids, _ = tokenize_with_offsets(tokenizer, variant)
        if target_ids:
            n_target = min(len(target_ids), n_text_tokens)
            return list(range(n_text_tokens - n_target, n_text_tokens)), f"suffix_tokenized:{n_target}"

    return [n_text_tokens - 1], "suffix_last_token"
