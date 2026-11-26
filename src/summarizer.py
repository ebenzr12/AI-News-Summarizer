"""
AI News Summarizer - Summarizer Inference Pipeline
=================================================
Modular end-to-end summarization pipeline supporting multiple engines:
1. 'abstractive': Locally fine-tuned mT5-small Seq2Seq model (models/fine_tuned).
2. 'llm': Optional cloud LLM API (Gemini / OpenAI-compatible) via env vars.
3. 'extractive': Deterministic bilingual TF-IDF + Lead-Bias + MMR engine (always ready).
4. 'auto': Automatically picks fine_tuned mT5 -> LLM API -> extractive fallback.

Preserves original article language and facts. Summarization and translation
remain strictly decoupled processes.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    MAX_INPUT_LENGTH,
    GEN_NUM_BEAMS,
    GEN_MAX_LENGTH,
    GEN_MIN_LENGTH,
    GEN_LENGTH_PENALTY,
    GEN_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_SUMMARY_ENGINE,
    SUMMARY_LENGTH_PRESETS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)
from src.language_detector import (
    detect_language,
    get_language_display_name,
    normalize_language_code,
)
from src.preprocessing import clean_news_text, add_task_prefix, compute_text_stats
from src.extractive_summarizer import ExtractiveNewsSummarizer
from src.model_loader import is_fine_tuned_model_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Raised when a requested model/engine (e.g. fine-tuned mT5 or LLM) is not configured."""
    pass


class SummarizationError(RuntimeError):
    """Raised when summarization inference fails unexpectedly."""
    pass


class NewsSummarizer:
    """
    Bilingual News Summarizer supporting Indonesian and English articles
    across pluggable inference engines.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device=None,
        engine: str = DEFAULT_SUMMARY_ENGINE,
        allow_pretrained_fallback: bool = False,
    ):
        self.default_engine = engine
        self.allow_pretrained_fallback = allow_pretrained_fallback
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self._extractive = ExtractiveNewsSummarizer()

        if self.model is not None and self.tokenizer is not None:
            import torch
            from src.model_loader import get_device

            self.device = self.device or get_device()
            self.model = self.model.to(self.device)
            self.model.eval()
            self.default_engine = "abstractive"

    def _ensure_seq2seq_loaded(self):
        """Lazily load fine-tuned mT5 model when abstractive inference is invoked."""
        if self.model is not None and self.tokenizer is not None:
            return

        if not is_fine_tuned_model_available() and not self.allow_pretrained_fallback:
            raise ModelNotReadyError(
                "Model abstractive fine-tuned belum tersedia di 'models/fine_tuned/'. "
                "Jalankan 'python train.py --sample_id 1000 --sample_en 1000 --epochs 3' "
                "atau gunakan engine 'auto' / 'extractive'."
            )

        from src.model_loader import load_model, get_device

        self.device = self.device or get_device()
        model_type = "fine_tuned" if is_fine_tuned_model_available() else "pretrained"
        self.model, self.tokenizer = load_model(model_type=model_type, device=self.device)
        self.model.eval()

    @staticmethod
    def is_llm_configured() -> bool:
        return bool(GEMINI_API_KEY or OPENAI_API_KEY)

    def get_capabilities(self) -> Dict[str, Any]:
        """Return availability status for every summarization engine."""
        fine_tuned_ready = is_fine_tuned_model_available() or (self.model is not None)
        llm_ready = self.is_llm_configured()
        return {
            "extractive": True,
            "abstractive_mt5": fine_tuned_ready,
            "llm_api": llm_ready,
            "active_auto_engine": (
                "abstractive"
                if fine_tuned_ready
                else ("llm" if llm_ready else "extractive")
            ),
        }

    def _resolve_engine(self, requested_engine: Optional[str]) -> str:
        eng = (requested_engine or self.default_engine or "auto").strip().lower()
        if eng not in ("auto", "extractive", "abstractive", "llm"):
            raise ValueError(
                f"Engine '{eng}' tidak valid. Pilih: 'auto', 'extractive', 'abstractive', atau 'llm'."
            )
        if eng == "auto":
            if self.model is not None or is_fine_tuned_model_available():
                return "abstractive"
            if self.is_llm_configured():
                return "llm"
            return "extractive"
        return eng

    def _summarize_abstractive(
        self,
        cleaned_text: str,
        lang_code: str,
        length: str,
        num_beams: int,
        max_length: Optional[int],
        min_length: Optional[int],
        length_penalty: float,
    ) -> str:
        import torch

        self._ensure_seq2seq_loaded()
        preset = SUMMARY_LENGTH_PRESETS.get(length, SUMMARY_LENGTH_PRESETS["balanced"])
        eff_max_len = max_length or preset["max_tokens"]
        eff_min_len = min_length or preset["min_tokens"]

        prefixed_input = add_task_prefix(cleaned_text, lang_code)
        inputs = self.tokenizer(
            prefixed_input,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                num_beams=num_beams,
                max_length=eff_max_len,
                min_length=eff_min_len,
                length_penalty=length_penalty,
                no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM_SIZE,
                early_stopping=True,
            )

        decoded = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return clean_news_text(decoded)

    def _summarize_llm(self, cleaned_text: str, lang_code: str, length: str) -> str:
        """Call external Gemini or OpenAI-compatible REST API when explicitly configured."""
        lang_label = "Bahasa Indonesia" if lang_code == "id" else "English"
        preset = SUMMARY_LENGTH_PRESETS.get(length, SUMMARY_LENGTH_PRESETS["balanced"])
        sent_range = f"{preset['min_sentences']}-{preset['max_sentences']} sentences"

        system_prompt = (
            f"You are a senior news editor. Summarize the following news article in {lang_label} ONLY. "
            f"Do NOT translate. Preserve all factual numbers, dates, and named entities accurately. "
            f"Keep the summary concise ({sent_range}). Return only the summary text."
        )

        try:
            if GEMINI_API_KEY:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
                )
                payload = {
                    "contents": [
                        {"role": "user", "parts": [{"text": f"{system_prompt}\n\nArticle:\n{cleaned_text}"}]}
                    ],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300},
                }
                resp = requests.post(url, json=payload, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                text_out = data["candidates"][0]["content"]["parts"][0]["text"]
                return clean_news_text(text_out)

            if OPENAI_API_KEY:
                url = f"{OPENAI_BASE_URL}/chat/completions"
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
                payload = {
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": cleaned_text},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 300,
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                text_out = data["choices"][0]["message"]["content"]
                return clean_news_text(text_out)
        except requests.RequestException as exc:
            raise SummarizationError(f"LLM API request failed: {exc}") from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise SummarizationError(f"Unexpected LLM API response format: {exc}") from exc

        raise ModelNotReadyError(
            "LLM engine diminta tetapi GEMINI_API_KEY atau OPENAI_API_KEY belum dikonfigurasi."
        )

    def summarize(
        self,
        text: str,
        language: str = "auto",
        length: str = "balanced",
        engine: Optional[str] = None,
        num_beams: int = GEN_NUM_BEAMS,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        length_penalty: float = GEN_LENGTH_PENALTY,
    ) -> Dict[str, Any]:
        """
        Summarize a news article in its original language.
        """
        start_time = time.perf_counter()

        cleaned_text = clean_news_text(text)
        if not cleaned_text:
            raise ValueError("Teks artikel tidak boleh kosong.")

        norm_lang = normalize_language_code(language)
        if norm_lang == "auto":
            lang_code, confidence = detect_language(cleaned_text)
        else:
            lang_code = norm_lang
            confidence = 1.0

        lang_name = get_language_display_name(lang_code)

        length_key = (length or "balanced").strip().lower()
        if length_key not in SUMMARY_LENGTH_PRESETS:
            raise ValueError(
                f"Preset panjang '{length}' tidak valid. Pilih: 'concise', 'balanced', atau 'detailed'."
            )

        resolved_engine = self._resolve_engine(engine)

        if resolved_engine == "abstractive":
            cleaned_summary = self._summarize_abstractive(
                cleaned_text,
                lang_code=lang_code,
                length=length_key,
                num_beams=num_beams,
                max_length=max_length or GEN_MAX_LENGTH,
                min_length=min_length or GEN_MIN_LENGTH,
                length_penalty=length_penalty,
            )
        elif resolved_engine == "llm":
            cleaned_summary = self._summarize_llm(cleaned_text, lang_code, length_key)
        else:
            cleaned_summary = self._extractive.summarize(
                cleaned_text, language=lang_code, length=length_key
            )

        input_stats = compute_text_stats(cleaned_text)
        summary_stats = compute_text_stats(cleaned_summary)

        input_word_count = input_stats["word_count"]
        summary_word_count = summary_stats["word_count"]
        compression_ratio = (
            max(0.0, (1.0 - (summary_word_count / input_word_count)) * 100)
            if input_word_count > 0
            else 0.0
        )
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "summary": cleaned_summary,
            "detected_language": lang_code,
            "language_name": lang_name,
            "confidence": confidence,
            "input_word_count": input_word_count,
            "summary_word_count": summary_word_count,
            "input_sentence_count": input_stats["sentence_count"],
            "summary_sentence_count": summary_stats["sentence_count"],
            "compression_ratio": f"{compression_ratio:.1f}%",
            "compression_value": round(compression_ratio, 1),
            "engine_used": resolved_engine,
            "length_preset": length_key,
            "latency_ms": latency_ms,
        }
