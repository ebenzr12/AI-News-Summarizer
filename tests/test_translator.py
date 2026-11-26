import unittest
from unittest.mock import patch

from src.translator import (
    NewsTranslator,
    TranslationError,
    chunk_text_for_translation,
)


class TestTranslator(unittest.TestCase):
    def test_chunk_text_for_translation_respects_max_chars(self):
        long_text = " ".join(
            [f"Kalimat berita nomor {i} membahas ekonomi digital nasional dengan rinci." for i in range(1, 20)]
        )
        chunks = chunk_text_for_translation(long_text, max_chars=200)
        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)

    def test_same_language_noop(self):
        translator = NewsTranslator()
        res = translator.translate(
            "Pemerintah mempercepat pembangunan transportasi massal.",
            source_lang="id",
            target_lang="id",
        )
        self.assertEqual(res["provider_used"], "noop_same_language")
        self.assertEqual(res["source_language"], "id")
        self.assertEqual(res["target_language"], "id")

    def test_fallback_from_mymemory_to_google_and_cache(self):
        translator = NewsTranslator()
        with patch.object(
            translator, "_translate_with_mymemory", side_effect=RuntimeError("timeout")
        ) as mock_mm, patch.object(
            translator,
            "_translate_with_google",
            return_value="The government is accelerating mass transit construction.",
        ) as mock_gg:
            first = translator.translate(
                "Pemerintah mempercepat pembangunan transportasi massal.",
                source_lang="id",
                target_lang="en",
            )
            self.assertEqual(first["provider_used"], "google_translate")
            self.assertFalse(first["cached"])
            self.assertEqual(mock_mm.call_count, 1)
            self.assertEqual(mock_gg.call_count, 1)

            second = translator.translate(
                "Pemerintah mempercepat pembangunan transportasi massal.",
                source_lang="id",
                target_lang="en",
            )
            self.assertTrue(second["cached"])
            self.assertEqual(mock_gg.call_count, 1)

    def test_all_providers_fail_raises_translation_error(self):
        translator = NewsTranslator()
        with patch.object(
            translator, "_translate_with_mymemory", side_effect=RuntimeError("offline")
        ), patch.object(
            translator, "_translate_with_google", side_effect=RuntimeError("offline")
        ):
            with self.assertRaises(TranslationError):
                translator.translate(
                    "The government confirmed new solar funding.",
                    source_lang="en",
                    target_lang="id",
                )


if __name__ == "__main__":
    unittest.main()
