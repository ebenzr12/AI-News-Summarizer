"""
Application service layer coordinating summarization, translation,
language detection, sample news articles, and system health reporting.
"""

from typing import Dict, List, Optional
import torch

from config.config import MIN_ARTICLE_WORDS
from backend.errors import InputValidationError
from backend.schemas import (
    DetectLanguageRequest,
    DetectLanguageResponse,
    HealthResponse,
    SampleArticle,
    SummarizeRequest,
    SummarizeResponse,
    TranslateRequest,
    TranslateResponse,
)
from src.language_detector import detect_language, get_language_display_name
from src.preprocessing import clean_news_text, compute_text_stats
from src.summarizer import NewsSummarizer
from src.translator import NewsTranslator


SAMPLE_ARTICLES: List[Dict[str, str]] = [
    {
        "id": "id-mrt-jakarta",
        "title": "Ekspansi Transportasi Publik MRT Jakarta & Pengurangan Emisi",
        "language": "id",
        "language_name": "Bahasa Indonesia",
        "category": "Nasional & Infrastruktur",
        "text": (
            "Pemerintah Provinsi DKI Jakarta bersama Kementerian Perhubungan mempercepat pembangunan "
            "koridor Moda Raya Terpadu (MRT) Fase 2A yang menghubungkan Bundaran Hotel Indonesia hingga Kota Tua. "
            "Proyek strategis nasional sepanjang 5,8 kilometer ini ditargetkan beroperasi penuh secara bertahap mulai 2027 "
            "dengan kapasitas angkut mencapai 350 ribu penumpang per hari. "
            "Kepala Dinas Perhubungan menyatakan bahwa integrasi tarif antarmoda dengan TransJakarta dan KRL Commuter Line "
            "menjadi kunci mendorong peralihan warga dari kendaraan pribadi ke angkutan umum. "
            "Berdasarkan kajian kualitas udara tahunan, peningkatan pengguna transportasi massal berbasis rel sebesar 25 persen "
            "berpotensi menekan emisi karbon sektor darat hingga 18 persen di wilayah Jabodetabek. "
            "Selain membangun tujuh stasiun bawah tanah baru, pemerintah juga menyiapkan kawasan berorientasi transit "
            "di sekitar Dukuh Atas dan Harmoni untuk mendukung mobilitas pejalan kaki dan ekonomi lokal."
        ),
    },
    {
        "id": "id-ai-ekonomi-digital",
        "title": "Adopsi Kecerdasan Buatan di Sektor UMKM dan Perbankan Indonesia",
        "language": "id",
        "language_name": "Bahasa Indonesia",
        "category": "Teknologi & Ekonomi",
        "text": (
            "Asosiasi Ekonomi Digital Indonesia melaporkan bahwa adopsi teknologi kecerdasan buatan (AI) di kalangan "
            "usaha mikro, kecil, dan menengah (UMKM) serta lembaga keuangan meningkat sebesar 42 persen sepanjang tahun ini. "
            "Penerapan AI paling banyak dimanfaatkan untuk otomasi layanan pelanggan dwibahasa, analisis risiko kredit mikro, "
            "serta prediksi permintaan rantai pasok di kota-kota tingkat kedua. "
            "Bank Indonesia bersama Otoritas Jasa Keuangan turut menerbitkan pedoman tata kelola AI yang menekankan "
            "transparansi algoritma, perlindungan data pribadi nasabah, dan audit keamanan siber secara berkala. "
            "Para pengamat menilai kolaborasi antara perguruan tinggi, perusahaan rintisan lokal, dan regulator "
            "akan memperkuat kedaulatan talenta digital nasional dalam menghadapi persaingan ekonomi Asia Tenggara."
        ),
    },
    {
        "id": "en-esa-solar-probe",
        "title": "European Solar Observatory Expands Space Weather Monitoring",
        "language": "en",
        "language_name": "English",
        "category": "Science & Space",
        "text": (
            "The European Space Agency has confirmed a two-year mission extension for its flagship solar observation satellite "
            "following unprecedented measurements of the Sun's polar magnetic fields. "
            "Equipped with ten specialized remote-sensing and in-situ instruments, the spacecraft has transmitted more than "
            "40 terabytes of high-resolution coronal imagery back to ground stations in Madrid and Darmstadt. "
            "Astrophysicists say the data is critical for improving early-warning forecasts of severe geomagnetic storms "
            "that can disrupt satellite navigation, aviation communications, and regional power grids. "
            "In coordination with NASA's Parker Solar Probe, researchers will now map solar wind acceleration across "
            "higher orbital inclinations ahead of the next solar activity cycle."
        ),
    },
    {
        "id": "en-southeast-asia-grid",
        "title": "ASEAN Grid Interconnection Accelerates Clean Energy Transition",
        "language": "en",
        "language_name": "English",
        "category": "Energy & Policy",
        "text": (
            "Energy ministers across Southeast Asia have signed a framework agreement to expand cross-border electricity "
            "interconnectors capable of transmitting 6 gigawatts of renewable power by 2032. "
            "The multilateral initiative links hydroelectric, geothermal, and offshore wind corridors across Indonesia, "
            "Malaysia, Singapore, and Thailand to balance seasonal fluctuations in energy demand. "
            "According to the Asian Development Bank, harmonized grid standards and joint green-financing mechanisms "
            "could cut regional power generation costs by 14 percent while avoiding 28 million metric tons of carbon emissions annually. "
            "Pilot subsea cable surveys in the Strait of Malacca are scheduled to begin in the third quarter, "
            "backed by a blended public-private investment fund."
        ),
    },
]


class SummarizerService:
    """Facade coordinating AI summarization, translation, and diagnostics."""

    def __init__(
        self,
        summarizer: Optional[NewsSummarizer] = None,
        translator: Optional[NewsTranslator] = None,
    ):
        self._summarizer = summarizer
        self._translator = translator

    @property
    def summarizer(self) -> NewsSummarizer:
        if self._summarizer is None:
            self._summarizer = NewsSummarizer()
        return self._summarizer

    @property
    def translator(self) -> NewsTranslator:
        if self._translator is None:
            self._translator = NewsTranslator()
        return self._translator

    def get_health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            cuda_available=bool(torch.cuda.is_available()),
            engines=self.summarizer.get_capabilities(),
            supported_languages=["id", "en"],
        )

    def list_samples(self) -> List[SampleArticle]:
        return [SampleArticle(**item) for item in SAMPLE_ARTICLES]

    def detect_language_and_stats(self, req: DetectLanguageRequest) -> DetectLanguageResponse:
        cleaned = clean_news_text(req.text)
        lang_code, confidence = detect_language(cleaned)
        stats = compute_text_stats(cleaned)
        return DetectLanguageResponse(
            language=lang_code,  # type: ignore[arg-type]
            language_name=get_language_display_name(lang_code),
            confidence=confidence,
            word_count=stats["word_count"],
            char_count=stats["char_count"],
            sentence_count=stats["sentence_count"],
            reading_time_seconds=stats["reading_time_seconds"],
        )

    def summarize(self, req: SummarizeRequest) -> SummarizeResponse:
        cleaned = clean_news_text(req.text)
        word_count = len(cleaned.split())
        if word_count < MIN_ARTICLE_WORDS:
            raise InputValidationError(
                f"Artikel terlalu pendek ({word_count} kata). Harap masukkan minimal {MIN_ARTICLE_WORDS} kata agar ringkasan bermakna."
            )

        result = self.summarizer.summarize(
            cleaned,
            language=req.language,
            length=req.length,
            engine=req.engine,
        )
        return SummarizeResponse(**result)

    def translate(self, req: TranslateRequest) -> TranslateResponse:
        cleaned = clean_news_text(req.text)
        result = self.translator.translate(
            cleaned,
            target_lang=req.target_language,
            source_lang=req.source_language,
        )
        return TranslateResponse(**result)
