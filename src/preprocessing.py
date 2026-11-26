"""
AI News Summarizer - Text Preprocessing Module
==============================================
Handles text cleaning, task prefix addition, and dataset tokenization
formatting for Google mT5-small Seq2Seq model.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path for direct imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    TASK_PREFIX_INDONESIAN,
    TASK_PREFIX_ENGLISH,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
)


# Common abbreviations in Indonesian & English news that should not trigger sentence breaks
_ABBREVIATIONS = (
    "Dr", "Prof", "Ir", "H", "Hj", "Dra", "Drs", "Jl", "No", "PT", "CV", "Tbk",
    "Mr", "Mrs", "Ms", "Sr", "Jr", "St", "vs", "etc", "e.g", "i.e", "U.S", "U.K",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
    "Agust", "Okt", "Des", "Rp", "hal", "hlm", "dkk", "dll", "dsb", "dst",
)
_ABBR_PLACEHOLDER = "\u0000"
_DECIMAL_PLACEHOLDER = "\u0001"


def clean_news_text(text: str) -> str:
    """
    Clean and normalize news article text.
    Preserves all numbers, punctuation, dates, named entities, and case.

    Args:
        text (str): Raw article or summary string.

    Returns:
        str: Cleaned, normalized string.
    """
    if not isinstance(text, str):
        return ""

    # Replace multiple whitespaces, tabs, and carriage returns with a single space
    cleaned = re.sub(r"[\r\t\f\v]+", " ", text)
    # Collapse repeated spaces within lines
    cleaned = re.sub(r" {2,}", " ", cleaned)
    # Replace multiple consecutive newlines with single newline
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    # Strip leading and trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


def split_sentences(text: str) -> List[str]:
    """
    Split Indonesian or English news text into sentences while preserving
    abbreviations, initials, and decimal numbers.
    """
    cleaned = clean_news_text(text)
    if not cleaned:
        return []

    protected = cleaned
    # Protect decimal numbers like 12.5 or 1.000
    protected = re.sub(r"(?<=\d)\.(?=\d)", _DECIMAL_PLACEHOLDER, protected)

    # Protect known abbreviations followed by a period
    abbr_pattern = r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\."
    protected = re.sub(
        abbr_pattern,
        lambda m: m.group(1) + _ABBR_PLACEHOLDER,
        protected,
        flags=re.IGNORECASE,
    )

    # Split on sentence-ending punctuation followed by whitespace/newline or end
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", protected)

    sentences: List[str] = []
    for part in raw_parts:
        restored = (
            part.replace(_ABBR_PLACEHOLDER, ".")
            .replace(_DECIMAL_PLACEHOLDER, ".")
            .strip()
        )
        if restored:
            sentences.append(restored)

    return sentences


def compute_text_stats(text: str) -> Dict[str, int]:
    """
    Compute lightweight text statistics for UI and API responses.
    """
    cleaned = clean_news_text(text)
    if not cleaned:
        return {
            "word_count": 0,
            "char_count": 0,
            "sentence_count": 0,
            "reading_time_seconds": 0,
        }

    words = cleaned.split()
    sentences = split_sentences(cleaned)
    word_count = len(words)
    # Average reading speed ~200 words/minute -> 3.33 words/sec
    reading_seconds = max(1, round(word_count / 3.33))

    return {
        "word_count": word_count,
        "char_count": len(cleaned),
        "sentence_count": len(sentences),
        "reading_time_seconds": reading_seconds,
    }


def add_task_prefix(text: str, language: str) -> str:
    """
    Prepend the language-specific prefix required by the mT5 summarization model.

    Args:
        text (str): Input news article text.
        language (str): 'id'/'indonesian' or 'en'/'english'.

    Returns:
        str: Text with task prefix prepended.
    """
    lang = language.lower().strip()
    if lang in ("id", "indonesian"):
        prefix = TASK_PREFIX_INDONESIAN
    elif lang in ("en", "english"):
        prefix = TASK_PREFIX_ENGLISH
    else:
        raise ValueError(
            f"Bahasa '{language}' tidak didukung. Gunakan 'id'/'indonesian' atau 'en'/'english'."
        )

    # Prevent duplicate prefixes if already present
    if text.startswith(prefix):
        return text

    return f"{prefix}{text}"


def preprocess_batch_for_seq2seq(
    batch: Dict[str, List[Any]],
    tokenizer: Any,
    max_input_length: int = MAX_INPUT_LENGTH,
    max_target_length: int = MAX_TARGET_LENGTH,
) -> Dict[str, List[Any]]:
    """
    Preprocesses a batch of samples from the XL-Sum dataset for mT5 Seq2Seq fine-tuning.

    Args:
        batch (Dict[str, List[Any]]): Batch containing 'text', 'summary', and 'language'.
        tokenizer: Hugging Face PreTrainedTokenizer (AutoTokenizer for mT5).
        max_input_length (int): Max token length for source article.
        max_target_length (int): Max token length for target summary.

    Returns:
        Dict[str, List[Any]]: Tokenized inputs, attention_mask, and labels (-100 for padding).
    """
    inputs = []
    targets = []

    for text, summary, lang in zip(batch["text"], batch["summary"], batch["language"]):
        clean_src = clean_news_text(text)
        clean_tgt = clean_news_text(summary)

        # Prepend task prefix
        prefixed_src = add_task_prefix(clean_src, lang)

        inputs.append(prefixed_src)
        targets.append(clean_tgt)

    # Tokenize input articles
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        # Note: Padding is left to dynamic DataCollatorForSeq2Seq to save VRAM
    )

    # Tokenize target summaries
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
