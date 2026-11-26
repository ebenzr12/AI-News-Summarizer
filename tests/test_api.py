import unittest
from unittest.mock import patch
from starlette.testclient import TestClient

from backend.app import create_app
from backend.services import SummarizerService
from src.summarizer import NewsSummarizer
from src.translator import NewsTranslator, TranslationError


class TestWebAPI(unittest.TestCase):
    def setUp(self):
        self.translator = NewsTranslator()
        self.summarizer = NewsSummarizer()
        self.service = SummarizerService(
            summarizer=self.summarizer,
            translator=self.translator,
        )
        self.app = create_app(service=self.service)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def test_frontend_and_static_assets_served(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("warta.ai", res.text)

        js_res = self.client.get("/static/js/app.js")
        self.assertEqual(js_res.status_code, 200)

    def test_health_and_samples(self):
        health = self.client.get("/api/v1/health")
        self.assertEqual(health.status_code, 200)
        self.assertIn("X-Request-ID", health.headers)
        self.assertIn("X-Process-Time-Ms", health.headers)
        data = health.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["supported_languages"], ["id", "en"])

        samples = self.client.get("/api/v1/samples")
        self.assertEqual(samples.status_code, 200)
        items = samples.json()["samples"]
        self.assertGreaterEqual(len(items), 2)
        langs = {item["language"] for item in items}
        self.assertEqual(langs, {"id", "en"})

    def test_detect_language_endpoint_and_errors(self):
        ok = self.client.post(
            "/api/v1/detect-language",
            json={
                "text": "Pemerintah Indonesia meningkatkan investasi transportasi publik di Jakarta tahun ini."
            },
        )
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["language"], "id")

        unsupported = self.client.post(
            "/api/v1/detect-language",
            json={
                "text": "La economía española creció un tres por ciento durante el último trimestre del año."
            },
        )
        self.assertEqual(unsupported.status_code, 422)
        self.assertEqual(unsupported.json()["error"]["code"], "UNSUPPORTED_LANGUAGE")

    def test_summarize_endpoint_valid_and_invalid(self):
        article = (
            "The European Space Agency has confirmed a two-year mission extension for its solar satellite. "
            "Equipped with ten specialized instruments, the spacecraft transmitted 40 terabytes of imagery. "
            "Scientists say the measurements improve early-warning forecasts for severe geomagnetic storms. "
            "Teams will coordinate closely with NASA across upcoming solar activity cycles."
        )
        res = self.client.post(
            "/api/v1/summarize",
            json={"text": article, "language": "auto", "length": "concise", "engine": "auto"},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["detected_language"], "en")
        self.assertEqual(body["engine_used"], "extractive")
        self.assertLess(body["summary_word_count"], body["input_word_count"])

        short_res = self.client.post(
            "/api/v1/summarize",
            json={"text": "Too short news", "language": "en", "length": "balanced"},
        )
        self.assertEqual(short_res.status_code, 422)
        self.assertEqual(short_res.json()["error"]["code"], "VALIDATION_ERROR")

        bad_json = self.client.post(
            "/api/v1/summarize",
            content="{not-json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(bad_json.status_code, 422)

    def test_translate_endpoint_success_and_outage(self):
        with patch.object(
            self.translator,
            "_translate_with_mymemory",
            return_value="Pemerintah mempercepat proyek transportasi massal di Jakarta.",
        ):
            res = self.client.post(
                "/api/v1/translate",
                json={
                    "text": "The government is accelerating mass transit projects in Jakarta.",
                    "source_language": "en",
                    "target_language": "id",
                },
            )
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["target_language"], "id")
            self.assertIn("Pemerintah", res.json()["translated_text"])

        with patch.object(
            self.translator,
            "translate",
            side_effect=TranslationError("Upstream unreachable"),
        ):
            outage = self.client.post(
                "/api/v1/translate",
                json={
                    "text": "Ringkasan berita ekonomi.",
                    "source_language": "id",
                    "target_language": "en",
                },
            )
            self.assertEqual(outage.status_code, 503)
            self.assertEqual(outage.json()["error"]["code"], "TRANSLATION_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
