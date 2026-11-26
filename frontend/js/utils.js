/**
 * Pure utility functions for text analysis, validation, and formatting.
 * Written as dependency-free ES modules for direct unit testing in Node.
 */

export const MIN_WORDS = 10;
export const MAX_CHARS = 50000;

export function computeClientStats(text) {
  const trimmed = (text || "").trim();
  if (!trimmed) {
    return { wordCount: 0, charCount: 0, readingSeconds: 0 };
  }
  const words = trimmed.split(/\s+/).filter(Boolean);
  const wordCount = words.length;
  const charCount = trimmed.length;
  const readingSeconds = Math.max(1, Math.round(wordCount / 3.33));
  return { wordCount, charCount, readingSeconds };
}

export function validateArticleInput(text, uiLang = "en") {
  const { wordCount, charCount } = computeClientStats(text);
  if (charCount === 0) {
    return {
      valid: false,
      message:
        uiLang === "id"
          ? "Harap masukkan teks berita terlebih dahulu."
          : "Please enter or paste a news article first.",
    };
  }
  if (wordCount < MIN_WORDS) {
    return {
      valid: false,
      message:
        uiLang === "id"
          ? `Artikel terlalu pendek (${wordCount} kata). Minimal ${MIN_WORDS} kata diperlukan.`
          : `Article is too short (${wordCount} words). At least ${MIN_WORDS} words are required.`,
    };
  }
  if (charCount > MAX_CHARS) {
    return {
      valid: false,
      message:
        uiLang === "id"
          ? `Artikel melebihi batas maksimal ${MAX_CHARS.toLocaleString()} karakter.`
          : `Article exceeds maximum limit of ${MAX_CHARS.toLocaleString()} characters.`,
    };
  }
  return { valid: true, message: "" };
}

export function getOppositeLanguage(langCode) {
  return langCode === "id" ? "en" : "id";
}

export function buildMarkdownReport({ articleText, summaryResult, translationResult }) {
  if (!summaryResult) return "";
  const lines = [
    "# AI News Summary Report",
    "",
    `- **Source Language:** ${summaryResult.language_name} (${summaryResult.detected_language.toUpperCase()})`,
    `- **Words:** ${summaryResult.input_word_count} -> ${summaryResult.summary_word_count} (${summaryResult.compression_ratio} compressed)`,
    `- **Engine:** ${summaryResult.engine_used} (${summaryResult.latency_ms} ms)`,
    "",
    "## Summary",
    "",
    summaryResult.summary,
  ];

  if (translationResult && translationResult.translated_text) {
    lines.push(
      "",
      `## Translation (${translationResult.target_name})`,
      "",
      translationResult.translated_text
    );
  }

  if (articleText) {
    lines.push("", "## Original Article Excerpt", "", articleText.trim().slice(0, 800));
  }

  return lines.join("\n");
}
