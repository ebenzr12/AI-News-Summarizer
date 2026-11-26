import unittest

from src.extractive_summarizer import ExtractiveNewsSummarizer
from src.summarizer import NewsSummarizer, ModelNotReadyError


INDONESIAN_ARTICLE = (
    "Pemerintah Provinsi DKI Jakarta mempercepat pembangunan jalur MRT Fase 2A dari Bundaran HI menuju Kota Tua. "
    "Proyek strategis sepanjang 5,8 kilometer ini ditargetkan melayani 350 ribu penumpang per hari pada tahun 2027. "
    "Dinas Perhubungan menekankan pentingnya integrasi tarif antara MRT, TransJakarta, dan KRL Commuter Line. "
    "Kajian lingkungan menunjukkan peralihan warga ke transportasi rel mampu menekan emisi karbon hingga 18 persen. "
    "Selain pembangunan tujuh stasiun bawah tanah, pemerintah menyiapkan kawasan transit terpadu di Harmoni. "
    "Langkah tersebut diharapkan mengurangi kemacetan kronis serta mendorong pertumbuhan ekonomi usaha kecil."
)

ENGLISH_ARTICLE = (
    "Southeast Asian energy ministers signed an agreement to expand cross-border renewable power interconnectors. "
    "The regional grid will transmit up to 6 gigawatts of clean electricity across four countries by 2032. "
    "Analysts at the Asian Development Bank estimate the project could cut generation costs by 14 percent annually. "
    "Subsea cable surveys in the Strait of Malacca are scheduled to start in the third quarter this year. "
    "Officials noted that shared hydroelectric and geothermal reserves will stabilize peak seasonal electricity demand. "
    "Private institutional investors have pledged initial financing for the first transmission corridor."
)


class TestExtractiveSummarizer(unittest.TestCase):
    def setUp(self):
        self.engine = ExtractiveNewsSummarizer()

    def test_indonesian_summary_is_concise_and_preserves_language(self):
        summary = self.engine.summarize(INDONESIAN_ARTICLE, language="id", length="concise")
        self.assertTrue(summary)
        self.assertLess(len(summary.split()), len(INDONESIAN_ARTICLE.split()))
        self.assertIn("Jakarta", summary)

    def test_english_length_presets_scale_appropriately(self):
        concise = self.engine.summarize(ENGLISH_ARTICLE, language="en", length="concise")
        detailed = self.engine.summarize(ENGLISH_ARTICLE, language="en", length="detailed")
        self.assertLessEqual(len(concise.split()), len(detailed.split()))

    def test_handles_single_sentence_and_stopwords_gracefully(self):
        single = "Berita singkat mengenai rapat kabinet hari ini."
        self.assertEqual(self.engine.summarize(single, language="id"), single)

        stop_only = "dan di ke dari. yang ini itu. atau juga."
        out = self.engine.summarize(stop_only, language="id", length="concise")
        self.assertTrue(len(out) > 0)

    def test_news_summarizer_auto_selects_extractive_when_no_finetuned_model(self):
        summarizer = NewsSummarizer()
        res = summarizer.summarize(ENGLISH_ARTICLE, language="auto", length="balanced", engine="auto")
        self.assertEqual(res["detected_language"], "en")
        self.assertEqual(res["engine_used"], "extractive")
        self.assertGreater(res["compression_value"], 0.0)
        self.assertLess(res["summary_word_count"], res["input_word_count"])

    def test_abstractive_raises_model_not_ready_without_weights(self):
        summarizer = NewsSummarizer()
        if not summarizer.get_capabilities()["abstractive_mt5"]:
            with self.assertRaises(ModelNotReadyError):
                summarizer.summarize(ENGLISH_ARTICLE, engine="abstractive")


if __name__ == "__main__":
    unittest.main()
