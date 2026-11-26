import test from "node:test";
import assert from "node:assert/strict";

import {
  buildMarkdownReport,
  computeClientStats,
  getOppositeLanguage,
  validateArticleInput,
} from "../frontend/js/utils.js";

test("computeClientStats handles empty and multi-word strings", () => {
  assert.deepEqual(computeClientStats("   "), {
    wordCount: 0,
    charCount: 0,
    readingSeconds: 0,
  });

  const stats = computeClientStats("One two three four five six seven.");
  assert.equal(stats.wordCount, 7);
  assert.equal(stats.charCount, 34);
  assert.ok(stats.readingSeconds >= 1);
});

test("validateArticleInput rejects empty and short articles", () => {
  const emptyRes = validateArticleInput("", "en");
  assert.equal(emptyRes.valid, false);

  const shortRes = validateArticleInput("Berita hari ini singkat", "id");
  assert.equal(shortRes.valid, false);
  assert.match(shortRes.message, /terlalu pendek/);

  const validRes = validateArticleInput(
    "Pemerintah mempercepat pembangunan jalur transportasi publik di Jakarta untuk menekan emisi.",
    "id"
  );
  assert.equal(validRes.valid, true);
});

test("getOppositeLanguage swaps id and en", () => {
  assert.equal(getOppositeLanguage("id"), "en");
  assert.equal(getOppositeLanguage("en"), "id");
});

test("buildMarkdownReport includes summary and optional translation", () => {
  const md = buildMarkdownReport({
    articleText: "Full article body.",
    summaryResult: {
      detected_language: "id",
      language_name: "Bahasa Indonesia",
      input_word_count: 50,
      summary_word_count: 15,
      compression_ratio: "70.0%",
      engine_used: "extractive",
      latency_ms: 5.2,
      summary: "Ringkasan berita utama.",
    },
    translationResult: {
      target_name: "English",
      translated_text: "Main news summary.",
    },
  });

  assert.match(md, /## Summary/);
  assert.match(md, /Ringkasan berita utama\./);
  assert.match(md, /## Translation \(English\)/);
  assert.match(md, /Main news summary\./);
});
