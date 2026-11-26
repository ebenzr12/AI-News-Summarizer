"""
AI News Summarizer - Language Detection Module
==============================================
Automatically detects whether the input news article is written in
Indonesian or English, validating against supported languages.
Combines statistical langdetect with bilingual lexical disambiguation
to avoid common misclassifications (e.g. Indonesian tagged as Malay).
"""

import logging
import re
from typing import Tuple, Set
from langdetect import detect_langs, DetectorFactory

# Set deterministic seed for reproducible language detection
DetectorFactory.seed = 42

logger = logging.getLogger(__name__)

# ISO 639-1 code mapping to standardized names
LANGUAGE_MAPPING = {
    "id": "indonesian",
    "en": "english",
}

SUPPORTED_CODES = set(LANGUAGE_MAPPING.keys())

# High-frequency function words used to disambiguate close Malay/Indonesian or short texts
_ID_MARKERS: Set[str] = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "pada", "dalam",
    "adalah", "ini", "itu", "oleh", "tidak", "akan", "atau", "juga", "telah",
    "sudah", "bahwa", "sebagai", "karena", "namun", "tetapi", "pemerintah",
    "masyarakat", "indonesia", "jakarta", "presiden", "menteri", "tersebut",
    "kepada", "antara", "secara", "bagi", "saat", "hingga", "melalui",
}

_EN_MARKERS: Set[str] = {
    "the", "and", "of", "to", "in", "is", "that", "for", "with", "on", "as",
    "are", "was", "were", "by", "this", "from", "at", "an", "be", "have",
    "has", "had", "not", "but", "which", "they", "their", "will", "would",
    "about", "into", "more", "than", "after", "over", "said", "government",
}

# Uniquely Indonesian news vocabulary vs Malay to resolve 'ms' confusion safely
_ID_STRONG_MARKERS: Set[str] = {
    "yang", "untuk", "dengan", "tidak", "pemerintah", "indonesia", "jakarta",
    "presiden", "menteri", "masyarakat", "tersebut", "karena", "namun", "tetapi",
    "sebagai", "bahwa", "melalui", "berdasarkan", "kabupaten", "provinsi", "rupiah",
}


class LanguageDetectionError(Exception):
    """Raised when language detection fails, is ambiguous, or language is unsupported."""
    pass


def normalize_language_code(lang: str) -> str:
    """
    Normalize a language specifier to 'id', 'en', or 'auto'.
    """
    code = (lang or "").strip().lower()
    if code in ("id", "indonesian", "bahasa indonesia", "indonesia"):
        return "id"
    if code in ("en", "english", "inggris"):
        return "en"
    if code == "auto":
        return "auto"
    raise ValueError(
        f"Bahasa '{lang}' tidak didukung. Gunakan 'id', 'en', atau 'auto'."
    )


def _lexical_score(text: str) -> Tuple[int, int]:
    """Count Indonesian vs English function-word markers in text."""
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    id_hits = sum(1 for t in tokens if t in _ID_MARKERS)
    en_hits = sum(1 for t in tokens if t in _EN_MARKERS)
    return id_hits, en_hits


def detect_language(text: str, min_confidence: float = 0.5) -> Tuple[str, float]:
    """
    Detect the language of the provided text.

    Args:
        text (str): Input news article text.
        min_confidence (float): Minimum probability threshold to log confidence warning.

    Returns:
        Tuple[str, float]: Standardized language code ('id' or 'en') and confidence score.

    Raises:
        ValueError: If text is empty or too short for reliable detection.
        LanguageDetectionError: If text is in an unsupported language or cannot be classified.
    """
    if not text or not text.strip():
        raise ValueError("Teks input kosong. Harap berikan teks artikel yang valid.")

    cleaned_text = text.strip()
    if len(cleaned_text.split()) < 3 and len(cleaned_text) < 15:
        raise ValueError(
            "Teks terlalu pendek untuk mendeteksi bahasa dengan akurat (minimal 3 kata)."
        )

    try:
        predictions = detect_langs(cleaned_text)
    except Exception as e:
        raise LanguageDetectionError(f"Gagal mendeteksi bahasa: {str(e)}")

    if not predictions:
        raise LanguageDetectionError("Tidak dapat mendeteksi bahasa dari teks yang diberikan.")

    top_pred = predictions[0]
    detected_code = top_pred.lang
    confidence = float(top_pred.prob)

    logger.info(f"Raw langdetect output: {predictions}")

    # Disambiguate Malay ('ms') or Tagalog ('tl') confusion when strong Indonesian markers exist
    if detected_code in ("ms", "tl"):
        tokens = set(re.findall(r"[a-zA-Z]+", cleaned_text.lower()))
        id_pred = next((p for p in predictions if p.lang == "id"), None)
        if id_pred is not None or len(tokens & _ID_STRONG_MARKERS) >= 2:
            id_hits, en_hits = _lexical_score(cleaned_text)
            if id_hits >= en_hits:
                detected_code = "id"
                confidence = max(confidence, id_pred.prob if id_pred else 0.75)

    # Disambiguate short mixed-loanword texts if a supported language is strongly evidenced lexically
    if detected_code not in SUPPORTED_CODES:
        id_hits, en_hits = _lexical_score(cleaned_text)
        total_words = max(1, len(cleaned_text.split()))
        # Only override if there is strong lexical evidence of ID or EN (>= 25% markers and >= 3 hits)
        if id_hits >= 3 and (id_hits / total_words) >= 0.2 and id_hits > en_hits:
            detected_code = "id"
            confidence = max(0.65, min(0.95, id_hits / total_words * 2))
        elif en_hits >= 3 and (en_hits / total_words) >= 0.2 and en_hits > id_hits:
            detected_code = "en"
            confidence = max(0.65, min(0.95, en_hits / total_words * 2))
        else:
            raise LanguageDetectionError(
                f"Bahasa terdeteksi adalah '{detected_code}', namun sistem hanya mendukung "
                f"Bahasa Indonesia ('id') dan Bahasa Inggris ('en')."
            )

    if confidence < min_confidence:
        logger.warning(
            f"Confidence {confidence:.2f} di bawah ambang batas {min_confidence:.2f}. "
            f"Tetap menggunakan bahasa teratas '{detected_code}'."
        )

    return detected_code, round(confidence, 4)


def get_language_display_name(lang_code: str) -> str:
    """
    Return human-readable language display name.
    """
    code = lang_code.lower().strip()
    if code in ("id", "indonesian"):
        return "Bahasa Indonesia"
    elif code in ("en", "english"):
        return "English"
    return code.upper()
