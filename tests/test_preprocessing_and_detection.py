import unittest

from src.preprocessing import (
    clean_news_text,
    split_sentences,
    compute_text_stats,
    add_task_prefix,
)
from src.language_detector import (
    detect_language,
    get_language_display_name,
    normalize_language_code,
    LanguageDetectionError,
)


class TestPreprocessing(unittest.TestCase):
    def test_clean_news_text_normalizes_whitespace_and_preserves_numbers(self):
        raw = "  Inflasi   tercatat\t\t3.5% pada 2026.\n\n\nBank Indonesia merilis laporan.  "
        cleaned = clean_news_text(raw)
        self.assertEqual(cleaned, "Inflasi tercatat 3.5% pada 2026.\nBank Indonesia merilis laporan.")

    def test_split_sentences_preserves_abbreviations_and_decimals(self):
        text = (
            "Dr. Budi Santoso dari PT. Nusantara menyatakan pertumbuhan mencapai 12.5 persen. "
            "Sementara itu, U.S. Federal Reserve menahan suku bunga di 4.25 persen."
        )
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertIn("Dr. Budi Santoso", sentences[0])
        self.assertIn("12.5", sentences[0])
        self.assertIn("U.S. Federal Reserve", sentences[1])

    def test_compute_text_stats(self):
        stats = compute_text_stats("Satu dua tiga empat. Lima enam tujuh delapan.")
        self.assertEqual(stats["word_count"], 8)
        self.assertEqual(stats["sentence_count"], 2)
        self.assertGreaterEqual(stats["reading_time_seconds"], 1)

    def test_add_task_prefix_no_duplicate(self):
        prefixed = add_task_prefix("Berita hari ini.", "id")
        self.assertTrue(prefixed.startswith("summarize Indonesian: "))
        self.assertEqual(add_task_prefix(prefixed, "id"), prefixed)


class TestLanguageDetector(unittest.TestCase):
    def test_detects_indonesian_article(self):
        text = (
            "Pemerintah Indonesia bersama Bank Indonesia meluncurkan program digitalisasi "
            "pembayaran untuk mempercepat transaksi ekonomi di seluruh provinsi."
        )
        code, conf = detect_language(text)
        self.assertEqual(code, "id")
        self.assertGreater(conf, 0.5)
        self.assertEqual(get_language_display_name(code), "Bahasa Indonesia")

    def test_detects_english_article(self):
        text = (
            "The European Space Agency confirmed a two-year extension for its solar "
            "observation mission to improve forecasts of geomagnetic storms."
        )
        code, conf = detect_language(text)
        self.assertEqual(code, "en")
        self.assertGreater(conf, 0.5)
        self.assertEqual(get_language_display_name(code), "English")

    def test_rejects_empty_or_too_short(self):
        with self.assertRaises(ValueError):
            detect_language("   ")
        with self.assertRaises(ValueError):
            detect_language("Halo")

    def test_rejects_unsupported_language(self):
        french_text = (
            "Le gouvernement français a annoncé aujourd'hui un nouveau plan d'investissement "
            "pour les énergies renouvelables et les transports ferroviaires."
        )
        with self.assertRaises(LanguageDetectionError):
            detect_language(french_text)

    def test_normalize_language_code(self):
        self.assertEqual(normalize_language_code("Indonesian"), "id")
        self.assertEqual(normalize_language_code("EN"), "en")
        self.assertEqual(normalize_language_code("auto"), "auto")
        with self.assertRaises(ValueError):
            normalize_language_code("fr")


if __name__ == "__main__":
    unittest.main()
