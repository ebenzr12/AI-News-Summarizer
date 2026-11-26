"""
Pydantic v2 request and response schemas for the AI News Summarizer API.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from config.config import MAX_ARTICLE_CHARS, MAX_TRANSLATE_CHARS


LanguageChoice = Literal["auto", "id", "en"]
LanguageCode = Literal["id", "en"]
SummaryLength = Literal["concise", "balanced", "detailed"]
EngineChoice = Literal["auto", "extractive", "abstractive", "llm"]


class DetectLanguageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_ARTICLE_CHARS)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Teks tidak boleh kosong.")
        return cleaned


class DetectLanguageResponse(BaseModel):
    language: LanguageCode
    language_name: str
    confidence: float
    word_count: int
    char_count: int
    sentence_count: int
    reading_time_seconds: int


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_ARTICLE_CHARS)
    language: LanguageChoice = "auto"
    length: SummaryLength = "balanced"
    engine: EngineChoice = "auto"

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Teks artikel tidak boleh kosong.")
        return cleaned


class SummarizeResponse(BaseModel):
    summary: str
    detected_language: LanguageCode
    language_name: str
    confidence: float
    input_word_count: int
    summary_word_count: int
    input_sentence_count: int
    summary_sentence_count: int
    compression_ratio: str
    compression_value: float
    engine_used: str
    length_preset: SummaryLength
    latency_ms: float


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TRANSLATE_CHARS)
    source_language: LanguageChoice = "auto"
    target_language: Optional[LanguageCode] = None

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Teks untuk diterjemahkan tidak boleh kosong.")
        return cleaned


class TranslateResponse(BaseModel):
    translated_text: str
    source_language: LanguageCode
    target_language: LanguageCode
    source_name: str
    target_name: str
    provider_used: str
    cached: bool
    latency_ms: float
    note: Optional[str] = None


class SampleArticle(BaseModel):
    id: str
    title: str
    language: LanguageCode
    language_name: str
    category: str
    text: str


class SamplesResponse(BaseModel):
    samples: List[SampleArticle]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    cuda_available: bool
    engines: Dict[str, object]
    supported_languages: List[LanguageCode]
