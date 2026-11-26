"""
AI News Summarizer - Translator Module
======================================
Independent translation module: separates summarization from translation.
Supports translating summaries or article text between Indonesian and English
with sentence-safe chunking, provider fallback, bounded caching, and structured errors.
"""

import sys
import time
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    TRANSLATION_MAX_CHUNK_CHARS,
    TRANSLATION_TIMEOUT_SECONDS,
    TRANSLATION_CACHE_SIZE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from src.language_detector import (
    detect_language,
    get_language_display_name,
    normalize_language_code,
)
from src.preprocessing import clean_news_text, split_sentences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    """Raised when all translation providers fail or are unreachable."""
    pass


def chunk_text_for_translation(
    text: str,
    max_chars: int = TRANSLATION_MAX_CHUNK_CHARS,
) -> List[str]:
    """
    Split text along sentence boundaries so each chunk stays <= max_chars.
    Falls back to word boundaries if a single sentence exceeds max_chars.
    """
    cleaned = clean_news_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = split_sentences(cleaned)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            # Split oversized sentence along word boundaries
            words = sentence.split()
            word_buf = ""
            for word in words:
                candidate = f"{word_buf} {word}".strip() if word_buf else word
                if len(candidate) <= max_chars:
                    word_buf = candidate
                else:
                    if word_buf:
                        chunks.append(word_buf)
                    word_buf = word[:max_chars] if len(word) > max_chars else word
            if word_buf:
                chunks.append(word_buf)
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


class NewsTranslator:
    """
    Independent Translator for Indonesian and English text.
    Kept completely decoupled from the summarization engine.
    """

    def __init__(self, cache_size: int = TRANSLATION_CACHE_SIZE):
        self._cache: OrderedDict[Tuple[str, str, str], Tuple[str, str]] = OrderedDict()
        self._cache_size = max(0, cache_size)

    def _get_cached(self, key: Tuple[str, str, str]) -> Optional[Tuple[str, str]]:
        if self._cache_size == 0 or key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _set_cached(self, key: Tuple[str, str, str], value: Tuple[str, str]) -> None:
        if self._cache_size == 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _translate_with_llm(self, text: str, src: str, tgt: str) -> str:
        """Translate using configured Gemini or OpenAI-compatible API."""
        src_name = get_language_display_name(src)
        tgt_name = get_language_display_name(tgt)
        prompt = (
            f"Translate the following news summary from {src_name} to {tgt_name}. "
            f"Preserve all factual names, numbers, dates, and journalistic tone. "
            f"Return only the translated text.\n\n{text}"
        )

        if GEMINI_API_KEY:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
            )
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512},
            }
            resp = requests.post(url, json=payload, timeout=TRANSLATION_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

        if OPENAI_API_KEY:
            url = f"{OPENAI_BASE_URL}/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            payload = {
                "model": OPENAI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 512,
            }
            resp = requests.post(
                url, json=payload, headers=headers, timeout=TRANSLATION_TIMEOUT_SECONDS
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        raise TranslationError("LLM API key not configured.")

    def _translate_with_mymemory(self, text: str, src: str, tgt: str) -> str:
        """Translate using MyMemoryTranslator in safe <=450-char chunks."""
        from deep_translator import MyMemoryTranslator

        src_tag = "id-ID" if src == "id" else "en-GB"
        tgt_tag = "en-GB" if tgt == "en" else "id-ID"

        translator = MyMemoryTranslator(source=src_tag, target=tgt_tag)
        chunks = chunk_text_for_translation(text)
        translated_chunks = []
        for chunk in chunks:
            res = translator.translate(chunk)
            if not res or not str(res).strip():
                raise TranslationError("MyMemory returned empty translation.")
            translated_chunks.append(str(res).strip())
        return " ".join(translated_chunks)

    def _translate_with_google(self, text: str, src: str, tgt: str) -> str:
        """Translate using GoogleTranslator fallback in safe chunks."""
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source=src, target=tgt)
        chunks = chunk_text_for_translation(text, max_chars=4000)
        translated_chunks = []
        for chunk in chunks:
            res = translator.translate(chunk)
            if not res or not str(res).strip():
                raise TranslationError("GoogleTranslator returned empty translation.")
            translated_chunks.append(str(res).strip())
        return " ".join(translated_chunks)

    def translate(
        self,
        text: str,
        target_lang: Optional[str] = None,
        source_lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Translate text between Indonesian and English.
        """
        start_time = time.perf_counter()
        cleaned_text = clean_news_text(text)
        if not cleaned_text:
            raise ValueError("Teks untuk diterjemahkan tidak boleh kosong.")

        # Determine source language
        if source_lang is None or str(source_lang).strip().lower() == "auto":
            detected_src, _ = detect_language(cleaned_text)
            source_lang = detected_src
        else:
            norm_src = normalize_language_code(source_lang)
            if norm_src == "auto":
                detected_src, _ = detect_language(cleaned_text)
                source_lang = detected_src
            else:
                source_lang = norm_src

        # Determine target language (opposite if not specified)
        if target_lang is None or str(target_lang).strip().lower() == "auto":
            target_lang = "en" if source_lang == "id" else "id"
        else:
            norm_tgt = normalize_language_code(target_lang)
            if norm_tgt == "auto":
                target_lang = "en" if source_lang == "id" else "id"
            else:
                target_lang = norm_tgt

        if source_lang == target_lang:
            return {
                "translated_text": cleaned_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "source_name": get_language_display_name(source_lang),
                "target_name": get_language_display_name(target_lang),
                "provider_used": "noop_same_language",
                "cached": False,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "note": "Bahasa sumber dan target sama, tidak ada penerjemahan.",
            }

        cache_key = (cleaned_text, source_lang, target_lang)
        cached_val = self._get_cached(cache_key)
        if cached_val is not None:
            translated_text, provider_used = cached_val
            return {
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "source_name": get_language_display_name(source_lang),
                "target_name": get_language_display_name(target_lang),
                "provider_used": provider_used,
                "cached": True,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        errors: List[str] = []
        translated_text = ""
        provider_used = ""

        providers = []
        if GEMINI_API_KEY or OPENAI_API_KEY:
            providers.append(("llm_api", self._translate_with_llm))
        providers.append(("mymemory", self._translate_with_mymemory))
        providers.append(("google_translate", self._translate_with_google))

        for name, fn in providers:
            try:
                raw_result = fn(cleaned_text, source_lang, target_lang)
                translated_text = clean_news_text(raw_result)
                if translated_text:
                    provider_used = name
                    break
            except Exception as exc:
                logger.warning(f"Translation provider '{name}' failed: {exc}")
                errors.append(f"{name}: {exc}")

        if not translated_text:
            raise TranslationError(
                "Layanan terjemahan saat ini tidak tersedia atau tidak dapat menjangkau jaringan. "
                f"Detail: {'; '.join(errors)}"
            )

        self._set_cached(cache_key, (translated_text, provider_used))

        return {
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang,
            "source_name": get_language_display_name(source_lang),
            "target_name": get_language_display_name(target_lang),
            "provider_used": provider_used,
            "cached": False,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }
