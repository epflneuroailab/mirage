"""Text feature extractor: LLaMA-3.2-3B word-level embeddings.

MIRAGE legacy ``LLAMA3p2`` extractor.

Architecture
------------
- Model: meta-llama/Llama-3.2-3B (28 transformer layers, hidden=3072)
- Feature frequency: 2.0 Hz stimulus grid
- Input: TSV transcript with columns words_per_tr / onsets_per_tr / durations_per_tr
- Output shape: (n_layers, n_dim, n_time)  — raw 2 Hz hidden states

The extractor is lazy — the model is loaded on first call to ``extract()``.
Layer pooling is not applied during extraction. The raw
2 Hz hidden states are cached, and pooling happens on the fly during
training/evaluation.
"""

from __future__ import annotations

import bisect
import logging
import typing as tp

import numpy as np
import torch

from brain_enc.features._alignment import overlap_slice
from brain_enc.features.base import (
    ExtractRequest,
    FeatureExtractor,
    FeatureOutput,
    build_extract_request,
    build_stimulus_paths_from_row,
    move_inputs_to_device,
    progress_iter,
    register,
    resolve_torch_dtype,
)
from brain_enc.features.multimodal.loaders import load_transcript_data
from brain_enc.modalities import Modality, conditioning_id

logger = logging.getLogger(__name__)

_MODEL_ID = "meta-llama/Llama-3.2-3B"
_FEATURE_HZ = 2.0   # 2 Hz stimulus grid from the reference pipeline
_SPACY_MODEL = "en_core_web_lg"


def _normalise_word(word: str) -> str:
    return word.lower().strip('",. ()?!\n\t')

@register
class LLaMA3p2Extractor(FeatureExtractor):
    """Word-level LLaMA-3.2-3B hidden-state extractor with TR alignment."""

    extractor_id = "llama3p2"
    modality: tp.Literal["text"] = "text"
    supported_target_modalities: tuple[Modality, ...] = ("text",)

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        max_context_len: int = 1024,
        batch_size: int = 8,
        device: str = "cpu",
        cache_dir: str | None = None,
        spacy_model: str = _SPACY_MODEL,
        allow_context_fallback: bool = False,
        max_unmatched_ratio: float = 0.05,
        available_modalities: tp.Iterable[str] | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        revision: str | None = None,
    ) -> None:
        self.model_id = model_id
        # Reference semantics: this is the number of context words.
        self.max_context_len = max_context_len
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = cache_dir
        self.spacy_model = spacy_model
        self.allow_context_fallback = allow_context_fallback
        if not (0.0 <= max_unmatched_ratio < 1.0):
            raise ValueError("max_unmatched_ratio must be >=0 and <1")
        self.max_unmatched_ratio = max_unmatched_ratio
        self._model = None
        self._tokenizer = None
        self._sentence_segmenter = None
        self.available_modalities = available_modalities
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.revision = revision

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading %s …", self.model_id)
        # truncation_side="left": when context exceeds max_length, keep the right
        # (most recent) portion so the target word is always at the sequence end.
        # Matches the legacy pipeline LLAMA3p2 tokenizer setup.
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, truncation_side="left", cache_dir=self.cache_dir
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # AutoModel (base transformer) instead of AutoModelForCausalLM — matches
        # the legacy implementation, which uses AutoModel for hidden-state extraction only.
        self._model = AutoModel.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            torch_dtype=resolve_torch_dtype(self.dtype),
        )
        self._model.to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def prepare(
        self,
        manifest_row: dict,
        *,
        target_modality: Modality | None = None,
        available_modalities: tuple[Modality, ...] | None = None,
    ) -> ExtractRequest:
        target_modality = target_modality or "text"
        if target_modality != "text":
            raise ValueError(f"{self.extractor_id} only supports target_modality='text'")
        return build_extract_request(
            item_id=manifest_row["stimulus_id"],
            target_modality="text",
            available_modalities=available_modalities or self.available_modalities,
            stimulus_paths=build_stimulus_paths_from_row(manifest_row),
            metadata=dict(manifest_row),
        )

    def extract(self, request: ExtractRequest) -> FeatureOutput:
        self._load_model()
        words, onsets, durations, total_duration = self._load_transcript(request.stimulus_path)
        if total_duration <= 0.0:
            return self._empty_output()

        # Embed each word with its left-context (all words seen so far).
        # Matches the legacy pipeline AddContextToWords + LLAMA3p2 which tokenizes the full
        # context string and extracts the last len(word) token positions.
        contexts, context_strategy = self._build_contexts(words)
        valid = [bool(context) for context in contexts]
        words = [word for word, keep in zip(words, valid) if keep]
        onsets = [onset for onset, keep in zip(onsets, valid) if keep]
        durations = [duration for duration, keep in zip(durations, valid) if keep]
        contexts = [context for context in contexts if context]
        if words:
            all_embeddings: list[np.ndarray] = []
            n_batches = (len(words) + self.batch_size - 1) // self.batch_size
            batch_iter = progress_iter(
                range(0, len(words), self.batch_size),
                desc=f"text words {request.item_id}",
                total=n_batches,
                leave=False,
                position=1,
            )
            for batch_start in batch_iter:
                batch_words = words[batch_start : batch_start + self.batch_size]
                batch_contexts = contexts[batch_start : batch_start + self.batch_size]
                all_embeddings.extend(self._embed_words_batch(batch_words, batch_contexts))
            stacked = np.stack(all_embeddings, axis=0)  # (n_words, n_layers, n_dim)
            n_layers, n_dim = stacked.shape[1], stacked.shape[2]
        else:
            stacked = np.zeros((0, 29, 3072), dtype=np.float32)
            n_layers, n_dim = stacked.shape[1], stacked.shape[2]

        # The legacy text path sums word embeddings into a 2 Hz TimedArray grid using the
        # TimedArray overlap/index rounding semantics. Unlike the reference
        # code, we keep the full layer stack and defer layer pooling.
        n_frames = max(1, round(total_duration * _FEATURE_HZ))
        text_2hz = np.zeros((n_layers, n_dim, n_frames), dtype=np.float32)
        out_duration = float(n_frames) / _FEATURE_HZ

        for i, (onset, duration) in enumerate(zip(onsets, durations)):
            sl = overlap_slice(
                out_start_s=0.0,
                out_duration_s=out_duration,
                word_start_s=float(onset),
                word_duration_s=float(duration),
                hz=_FEATURE_HZ,
                n_frames=n_frames,
            )
            if sl is None:
                continue
            text_2hz[:, :, sl] += stacked[i][:, :, None]

        time_axis = np.arange(n_frames, dtype=np.float32) / _FEATURE_HZ
        layer_axis = np.linspace(0.0, 1.0, n_layers, dtype=np.float32)

        return FeatureOutput(
            features=text_2hz,
            time_axis=time_axis,
            layer_axis=layer_axis,
            metadata={
                "model_id": self.model_id,
                "hf_model_id": self.model_id,
                "extractor_id": self.extractor_id,
                "target_modality": request.target_modality,
                "available_modalities": list(request.available_modalities),
                "conditioning_id": conditioning_id(request.available_modalities, target_modality="text"),
                "spacy_model": self.spacy_model,
                "n_words": len(words),
                "feature_hz": _FEATURE_HZ,
                "total_duration_s": total_duration,
                "context_strategy": context_strategy,
                "max_unmatched_ratio": self.max_unmatched_ratio,
                "layer_pooling_applied": False,
                "token_pooling": "mean_target_token_span",
                "temporal_aggregation": "sum_overlapping_words",
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_words_batch(
        self, words: list[str], contexts: list[str]
    ) -> list[np.ndarray]:
        """Embed a batch of words, each within its left-context string.

        Mirrors the legacy LLAMA3p2._get_data: tokenise the full batch with
        left-truncation + right-padding, run one forward pass, then for each
        item strip right-side padding and extract the last len(word) token
        positions as the word representation.

        Returns a list of (n_total_layers, n_dim) arrays, one per word.
        """
        inputs = self._tokenizer(
            contexts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            add_special_tokens=False,
        )
        inputs = move_inputs_to_device(inputs, self.device)
        pad_id = self._tokenizer.pad_token_id
        with torch.inference_mode():
            outputs = self._model(**inputs, output_hidden_states=True)
        # Include hidden_states[0] (embedding output) to match the legacy pipeline.
        hidden = torch.stack(outputs.hidden_states, dim=0)  # (n_layers+1, B, seq, H)

        results: list[np.ndarray] = []
        for i, word in enumerate(words):
            h = hidden[:, i]  # (n_layers+1, seq, H)
            # Strip right-side padding (tokenizer uses left-truncation, right-padding).
            n_pads = int((inputs["input_ids"][i] == pad_id).sum().item())
            if n_pads:
                h = h[:, :-n_pads]
            # Take last len(word) token positions; the legacy path uses string length as an
            # approximate token count so the target word tokens are at the tail.
            n_target = max(1, len(word))
            word_h = h[:, -n_target:]  # (n_layers+1, n_target, H)
            results.append(word_h.mean(dim=1).float().cpu().numpy())
        return results

    def _build_contexts(self, words: list[str]) -> tuple[list[str], str]:
        """Port the legacy text-enhancer chain used before LLAMA extraction."""
        running = self._build_running_contexts(words)
        enhancer_contexts = self._build_enhancer_contexts(words)
        if enhancer_contexts is None and self.allow_context_fallback:
            return running, "running_fallback"
        if enhancer_contexts is None:
            raise RuntimeError(
                "Failed to build legacy text contexts. "
                "Set allow_context_fallback=True to use running-context fallback."
            )
        return enhancer_contexts, "legacy_enhancers"

    def _build_running_contexts(self, words: list[str]) -> list[str]:
        contexts: list[str] = []
        prefix: list[str] = []
        max_words = None if self.max_context_len is None else self.max_context_len + 1
        for word in words:
            prefix.append(word)
            tokens = prefix[-max_words:] if max_words is not None else prefix
            contexts.append(" ".join(tokens))
        return contexts

    def _build_enhancer_contexts(self, words: list[str]) -> list[str] | None:
        if not words:
            return []

        punctuated_text = self._punctuate_text(words)
        if not punctuated_text:
            return None

        try:
            match_info = self._match_text_words(punctuated_text, words)
        except Exception:
            return None
        unmatched = sum(
            not (
                isinstance(item.get("sentence", None), str)
                and bool(item.get("sentence", None))
            )
            for item in match_info
        )
        ratio = unmatched / len(match_info) if match_info else 0.0
        if ratio > self.max_unmatched_ratio:
            raise RuntimeError(
                f"Ratio of unmatched words is {ratio:.4f} on {len(match_info)} words "
                f"while max_unmatched_ratio={self.max_unmatched_ratio}"
            )

        contexts: list[str] = []
        past_sentences: list[str] = []
        last_word: dict[str, tp.Any] | None = None
        for word, info in zip(words, match_info):
            sent = info.get("sentence", None)
            sent_char = info.get("sentence_char", None)
            if not (isinstance(sent, str) and sent):
                contexts.append("")
                last_word = None
                continue

            if sent_char is None or np.isnan(sent_char):
                contexts.append("")
                continue

            if last_word is not None:
                if sent != last_word["sentence"]:
                    if float(sent_char) <= float(last_word["sentence_char"]):
                        past_sentences.append(last_word["sentence"])

            last_char = float(sent_char) + len(word)
            context = "".join(past_sentences) + sent[: int(last_char)]
            if self.max_context_len is not None:
                context = " ".join(context.split(" ")[-self.max_context_len - 1 :])
            contexts.append(context)
            last_word = {"sentence": sent, "sentence_char": float(sent_char)}
        return contexts

    def _load_sentence_segmenter(self):
        if self._sentence_segmenter is not None:
            return self._sentence_segmenter
        try:
            import spacy
        except ImportError as exc:
            if self.allow_context_fallback:
                return None
            raise RuntimeError(
                "spaCy is required for legacy text context construction."
            ) from exc

        if not spacy.util.is_package(self.spacy_model):
            try:
                import spacy.cli

                spacy.cli.download(self.spacy_model)
            except Exception as exc:
                if self.allow_context_fallback:
                    return None
                raise RuntimeError(
                    f"Could not install spaCy model '{self.spacy_model}'."
                ) from exc
        nlp = spacy.load(self.spacy_model)
        self._sentence_segmenter = nlp
        return self._sentence_segmenter

    def _punctuate_text(self, words: list[str]) -> str:
        if not words:
            return ""
        nlp = self._load_sentence_segmenter()
        if nlp is None:
            return ""
        doc = nlp(" ".join(words))
        sentences = [sent.text.capitalize().rstrip(".") for sent in doc.sents if sent.text.strip()]
        if not sentences:
            return ""
        return ". ".join(sentences)

    def _match_text_words(
        self,
        text: str,
        words: list[str],
    ) -> list[dict[str, tp.Any]]:
        """Port the legacy AddSentenceToWords matching logic."""
        nlp = self._load_sentence_segmenter()
        if nlp is None:
            raise RuntimeError("Sentence segmenter unavailable")

        doc = nlp(text)
        text_words = [word for sentence in doc.sents for word in sentence]
        text_words_str = [_normalise_word(token.text) for token in text_words]
        text_match, words_match = _match_list(
            text_words_str,
            [_normalise_word(word) for word in words],
        )
        info: list[dict[str, tp.Any]] = [{"word": word} for word in words]
        match_key = "text_match"
        for tm, wm in zip(text_match, words_match):
            info[wm][match_key] = tm

        to_debug: list[dict[str, tp.Any]] = []
        first: dict[str, tp.Any] | None = None
        last: dict[str, tp.Any] | None = None
        for idx, item in enumerate(info):
            if match_key not in item:
                to_debug.append(item)
                if idx != len(info) - 1:
                    continue
            if match_key in item:
                last = item
            if to_debug:
                start = 0
                if first is not None:
                    token = text_words[first[match_key]]
                    start = token.idx + len(token)
                end = len(text)
                if last is not None:
                    token = text_words[last[match_key]]
                    end = token.idx
                subtext = text[start:end].lower()
                concat_words = " ".join(_normalise_word(entry["word"]) for entry in to_debug)
                sub_text_match, sub_word_match = _match_list(subtext, concat_words)
                word_idx_chars = [
                    (word_idx, char_idx)
                    for word_idx, entry in enumerate(to_debug)
                    for char_idx in range(len(entry["word"]) + 1)
                ]
                for match_text, match_wordseq in zip(sub_text_match, sub_word_match):
                    word_idx, char_idx = word_idx_chars[match_wordseq]
                    to_debug[word_idx].setdefault("votes", []).append(start + match_text - char_idx)
                for entry in to_debug:
                    if "votes" not in entry:
                        continue
                    votes: list[int] = entry.pop("votes")
                    best_bet = max(votes, key=votes.count)
                    if votes.count(best_bet) / max(1, len(entry["word"])) <= 0.5:
                        continue
                    bounds = [rec[match_key] if rec is not None else None for rec in [first, last]]
                    sub = text_words[bounds[0] : bounds[1]]
                    if not sub:
                        continue
                    ind = bisect.bisect_left(sub, best_bet, key=lambda token: token.idx + len(token))
                    ind = min(ind, len(sub) - 1)
                    entry["sentence"] = sub[ind].sent.text_with_ws
                    entry["sentence_char"] = best_bet - sub[ind].sent[0].idx
                to_debug = []

            if last is not None:
                first = last
                last = None

        prev_sentence: str | None = None
        missing: list[dict[str, tp.Any]] = []
        for entry in info:
            if match_key in entry:
                token = text_words[entry.pop(match_key)]
                entry["sentence_char"] = token.idx - token.sent[0].idx
                entry["sentence"] = token.sent.text_with_ws
            sent = entry.get("sentence", None)
            if sent is None:
                missing.append(entry)
                continue
            if prev_sentence == sent:
                for missing_entry in missing:
                    missing_entry["sentence"] = sent
            missing = []
            prev_sentence = sent

        for entry in info:
            entry.pop("word", None)
        return info

    def _load_transcript(
        self, path: str
    ) -> tuple[list[str], list[float], list[float], float]:
        transcript = load_transcript_data(path)
        return (
            transcript.words,
            transcript.onsets,
            transcript.durations,
            transcript.total_duration_s,
        )

    def _empty_output(self) -> FeatureOutput:
        return FeatureOutput(
            features=np.zeros((0, 3072, 1), dtype=np.float32),
            time_axis=np.array([0.0], dtype=np.float32),
            layer_axis=None,
            metadata={
                "model_id": self.model_id,
                "spacy_model": self.spacy_model,
                "n_words": 0,
                "feature_hz": _FEATURE_HZ,
                "max_unmatched_ratio": self.max_unmatched_ratio,
                "layer_pooling_applied": False,
                "token_pooling": "mean_target_token_span",
                "temporal_aggregation": "sum_overlapping_words",
            },
        )


def _match_list(
    a: list[str] | str,
    b: list[str] | str,
    on_replace: str = "delete",
) -> tuple[np.ndarray, np.ndarray]:
    """Local copy of the legacy Levenshtein-based sequence matcher."""
    from Levenshtein import editops

    if not isinstance(a, str):
        unique = np.unique(np.r_[a, b])
        label_encoder = {item: idx for idx, item in enumerate(unique)}

        def _int_to_unicode(array: np.ndarray) -> str:
            return "".join(str(chr(label_encoder[item])) for item in array)

        a = _int_to_unicode(np.asarray(a, dtype=object))
        b = _int_to_unicode(np.asarray(b, dtype=object))

    changes = editops(a, b)
    b_sel = np.arange(len(b)).astype(float)
    a_sel = np.arange(len(a)).astype(float)
    for change_type, val_a, val_b in changes:
        if change_type == "insert":
            b_sel[val_b] = np.nan
        elif change_type == "delete":
            a_sel[val_a] = np.nan
        elif on_replace == "delete":
            a_sel[val_a] = np.nan
            b_sel[val_b] = np.nan
        elif on_replace == "keep":
            continue
        else:
            raise NotImplementedError(f"Unknown on_replace={on_replace!r}")

    b_sel = b_sel[~np.isnan(b_sel)].astype(int)
    a_sel = a_sel[~np.isnan(a_sel)].astype(int)
    assert len(b_sel) == len(a_sel)
    return a_sel, b_sel
