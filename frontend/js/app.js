/**
 * Main UI controller for WartaAI News Summarizer & Translator.
 */

import { api } from "/static/js/api.js";
import { I18N } from "/static/js/i18n.js";
import { store } from "/static/js/state.js";
import {
  buildMarkdownReport,
  computeClientStats,
  getOppositeLanguage,
  validateArticleInput,
} from "/static/js/utils.js";

const els = {
  healthPill: document.getElementById("health-pill"),
  healthLabel: document.getElementById("health-label"),
  heroSampleBtn: document.getElementById("hero-sample-btn"),
  studioWorkspace: document.getElementById("studio-workspace"),
  themeToggleBtn: document.getElementById("theme-toggle-btn"),
  themeIcon: document.getElementById("theme-icon"),
  uiLangBtns: document.querySelectorAll("[data-ui-lang]"),
  sampleSelect: document.getElementById("sample-select"),
  fileInput: document.getElementById("file-input"),
  dropZone: document.getElementById("drop-zone"),
  articleInput: document.getElementById("article-input"),
  detectedLangPill: document.getElementById("detected-lang-pill"),
  clientStats: document.getElementById("client-stats"),
  clearBtn: document.getElementById("clear-btn"),
  sourceLangSelect: document.getElementById("source-lang-select"),
  lengthBtns: document.querySelectorAll("[data-length]"),
  engineSelect: document.getElementById("engine-select"),
  inputError: document.getElementById("input-error"),
  inputErrorTitle: document.getElementById("input-error-title"),
  inputErrorMsg: document.getElementById("input-error-msg"),
  summarizeBtn: document.getElementById("summarize-btn"),
  summarizeBtnLabel: document.getElementById("summarize-btn-label"),
  outputEmpty: document.getElementById("output-empty"),
  outputSkeleton: document.getElementById("output-skeleton"),
  summaryCard: document.getElementById("summary-card"),
  copySummaryBtn: document.getElementById("copy-summary-btn"),
  exportMdBtn: document.getElementById("export-md-btn"),
  metricLang: document.getElementById("metric-lang"),
  metricWords: document.getElementById("metric-words"),
  metricCompression: document.getElementById("metric-compression"),
  metricEngine: document.getElementById("metric-engine"),
  summaryLangBadge: document.getElementById("summary-lang-badge"),
  summaryConfidence: document.getElementById("summary-confidence"),
  summaryText: document.getElementById("summary-text"),
  targetLangSelect: document.getElementById("target-lang-select"),
  translateBtn: document.getElementById("translate-btn"),
  translateBtnLabel: document.getElementById("translate-btn-label"),
  translationError: document.getElementById("translation-error"),
  translationErrorMsg: document.getElementById("translation-error-msg"),
  retryTranslateBtn: document.getElementById("retry-translate-btn"),
  translationBox: document.getElementById("translation-box"),
  translationLangBadge: document.getElementById("translation-lang-badge"),
  translationProvider: document.getElementById("translation-provider"),
  translationText: document.getElementById("translation-text"),
  copyTranslationBtn: document.getElementById("copy-translation-btn"),
  historyToggleBtn: document.getElementById("history-toggle-btn"),
  historyDrawer: document.getElementById("history-drawer"),
  closeHistoryBtn: document.getElementById("close-history-btn"),
  clearHistoryBtn: document.getElementById("clear-history-btn"),
  historyList: document.getElementById("history-list"),
  historyCount: document.getElementById("history-count"),
  toast: document.getElementById("toast"),
};

let detectTimer = null;
let toastTimer = null;

function t(key) {
  return (I18N[store.uiLang] && I18N[store.uiLang][key]) || I18N.en[key] || key;
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    els.toast.hidden = true;
  }, 2400);
}

function applyTheme() {
  document.documentElement.setAttribute("data-theme", store.theme);
  els.themeIcon.textContent = store.theme === "dark" ? "Light" : "Dark";
}

function applyUiLanguage() {
  document.documentElement.lang = store.uiLang;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.getAttribute("data-i18n");
    if (!key) return;
    const text = t(key);
    if (key === "heroTitle") {
      const parts = text.split("\n");
      node.textContent = "";
      parts.forEach((part, idx) => {
        if (idx > 0) node.appendChild(document.createElement("br"));
        node.appendChild(document.createTextNode(part));
      });
    } else {
      node.textContent = text;
    }
  });
  els.uiLangBtns.forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.uiLang === store.uiLang);
  });
  if (!els.articleInput.value.trim()) {
    els.detectedLangPill.textContent = t("waitingText");
  }
  renderHistory();
}

function setButtonLoading(button, isLoading, labelEl, activeText, idleText) {
  button.disabled = isLoading;
  const spinner = button.querySelector(".btn-spinner");
  if (spinner) spinner.hidden = !isLoading;
  if (labelEl) labelEl.textContent = isLoading ? activeText : idleText;
}

function showInputError(message, title = "Input Error") {
  els.inputErrorTitle.textContent = title;
  els.inputErrorMsg.textContent = message;
  els.inputError.hidden = false;
}

function clearInputError() {
  els.inputError.hidden = true;
  els.inputErrorMsg.textContent = "";
}

function updateClientStats() {
  const { wordCount, charCount, readingSeconds } = computeClientStats(els.articleInput.value);
  const unitWords = store.uiLang === "id" ? "kata" : "words";
  const unitChars = store.uiLang === "id" ? "karakter" : "chars";
  const readStr = wordCount > 0 ? ` · ~${readingSeconds}s` : "";
  els.clientStats.textContent = `${wordCount} ${unitWords} · ${charCount} ${unitChars}${readStr}`;
}

function scheduleLanguageDetect() {
  clearTimeout(detectTimer);
  const text = els.articleInput.value.trim();

  if (!text) {
    els.detectedLangPill.className = "lang-pill lang-pill--idle";
    els.detectedLangPill.textContent = t("waitingText");
    return;
  }

  const { wordCount } = computeClientStats(text);
  if (wordCount < 3) {
    els.detectedLangPill.className = "lang-pill lang-pill--idle";
    els.detectedLangPill.textContent =
      store.uiLang === "id" ? "Ketik minimal 3 kata..." : "Type at least 3 words...";
    return;
  }

  detectTimer = setTimeout(async () => {
    try {
      const res = await api.detectLanguage(text);
      const pct = Math.round(res.confidence * 100);
      els.detectedLangPill.className = "lang-pill";
      els.detectedLangPill.textContent = `${res.language_name} (${pct}%)`;
    } catch (err) {
      els.detectedLangPill.className = "lang-pill lang-pill--warn";
      els.detectedLangPill.textContent = err.message;
    }
  }, 420);
}

function renderSummary(result) {
  store.summaryResult = result;
  els.outputEmpty.hidden = true;
  els.outputSkeleton.hidden = true;
  els.summaryCard.hidden = false;
  els.copySummaryBtn.disabled = false;
  els.exportMdBtn.disabled = false;

  els.metricLang.textContent = result.detected_language.toUpperCase();
  els.metricWords.textContent = `${result.input_word_count} → ${result.summary_word_count}`;
  els.metricCompression.textContent = result.compression_ratio;
  els.metricEngine.textContent = `${result.engine_used} · ${result.latency_ms}ms`;

  els.summaryLangBadge.textContent = result.language_name;
  els.summaryConfidence.textContent = `Confidence ${Math.round(result.confidence * 100)}%`;
  els.summaryText.textContent = result.summary;

  // Set default target language to the opposite of summary language
  els.targetLangSelect.value = getOppositeLanguage(result.detected_language);
}

function renderTranslation(result) {
  store.translationResult = result;
  els.translationError.hidden = true;
  els.translationBox.hidden = false;
  els.translationLangBadge.textContent = `${result.source_language.toUpperCase()} → ${result.target_name}`;
  const cacheTag = result.cached ? " · cached" : "";
  els.translationProvider.textContent = `${result.provider_used} · ${result.latency_ms}ms${cacheTag}`;
  els.translationText.textContent = result.translated_text;
}

function renderHistory() {
  els.historyCount.textContent = String(store.history.length);
  els.historyList.innerHTML = "";

  if (store.history.length === 0) {
    const empty = document.createElement("p");
    empty.className = "panel-desc";
    empty.textContent = t("historyEmpty");
    els.historyList.appendChild(empty);
    return;
  }

  store.history.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "history-item";

    const meta = document.createElement("div");
    meta.className = "history-item-meta";
    const langSpan = document.createElement("span");
    langSpan.textContent = `${item.summaryResult.language_name} · ${item.summaryResult.compression_ratio}`;
    const timeSpan = document.createElement("span");
    timeSpan.textContent = new Date(item.createdAt).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
    meta.append(langSpan, timeSpan);

    const preview = document.createElement("p");
    preview.className = "history-item-preview";
    preview.textContent = item.summaryResult.summary;

    btn.append(meta, preview);
    btn.addEventListener("click", () => {
      els.articleInput.value = item.articleText;
      updateClientStats();
      scheduleLanguageDetect();
      renderSummary(item.summaryResult);
      if (item.translationResult) {
        renderTranslation(item.translationResult);
      } else {
        els.translationBox.hidden = true;
        store.translationResult = null;
      }
      toggleHistoryDrawer(false);
    });

    els.historyList.appendChild(btn);
  });
}

function toggleHistoryDrawer(forceState) {
  const nextOpen =
    typeof forceState === "boolean" ? forceState : els.historyDrawer.hidden;
  els.historyDrawer.hidden = !nextOpen;
  els.historyToggleBtn.setAttribute("aria-expanded", String(nextOpen));
}

async function handleSummarize() {
  clearInputError();
  const rawText = els.articleInput.value;
  const check = validateArticleInput(rawText, store.uiLang);
  if (!check.valid) {
    showInputError(check.message);
    return;
  }

  store.translationResult = null;
  els.translationBox.hidden = true;
  els.translationError.hidden = true;
  els.outputEmpty.hidden = true;
  els.summaryCard.hidden = true;
  els.outputSkeleton.hidden = false;

  setButtonLoading(
    els.summarizeBtn,
    true,
    els.summarizeBtnLabel,
    t("summarizingBtn"),
    t("summarizeBtn")
  );

  try {
    const result = await api.summarize({
      text: rawText,
      language: els.sourceLangSelect.value,
      length: store.selectedLength,
      engine: els.engineSelect.value,
    });
    renderSummary(result);
    store.addHistory({
      articleText: rawText,
      summaryResult: result,
      translationResult: null,
    });
    renderHistory();
  } catch (err) {
    els.outputSkeleton.hidden = true;
    els.outputEmpty.hidden = !store.summaryResult;
    if (store.summaryResult) els.summaryCard.hidden = false;
    showInputError(err.message, err.code || "Error");
  } finally {
    setButtonLoading(
      els.summarizeBtn,
      false,
      els.summarizeBtnLabel,
      t("summarizingBtn"),
      t("summarizeBtn")
    );
  }
}

async function handleTranslate() {
  if (!store.summaryResult || !store.summaryResult.summary) return;
  els.translationError.hidden = true;

  setButtonLoading(
    els.translateBtn,
    true,
    els.translateBtnLabel,
    t("translatingBtn"),
    t("translateBtn")
  );

  try {
    const result = await api.translate({
      text: store.summaryResult.summary,
      sourceLanguage: store.summaryResult.detected_language,
      targetLanguage: els.targetLangSelect.value,
    });
    renderTranslation(result);
    if (store.history.length > 0 && store.history[0].summaryResult.summary === store.summaryResult.summary) {
      store.history[0].translationResult = result;
      renderHistory();
    }
  } catch (err) {
    els.translationBox.hidden = true;
    els.translationErrorMsg.textContent = err.message;
    els.translationError.hidden = false;
  } finally {
    setButtonLoading(
      els.translateBtn,
      false,
      els.translateBtnLabel,
      t("translatingBtn"),
      t("translateBtn")
    );
  }
}

function loadFileContent(file) {
  if (!file) return;
  if (file.size > 512 * 1024) {
    showInputError("File size exceeds 512 KB limit.");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    els.articleInput.value = String(reader.result || "");
    clearInputError();
    updateClientStats();
    scheduleLanguageDetect();
    showToast(t("fileLoadedToast"));
  };
  reader.readAsText(file);
}

async function init() {
  applyTheme();
  applyUiLanguage();

  // Load system health
  try {
    const health = await api.getHealth();
    const active = health.engines.active_auto_engine || "extractive";
    els.healthPill.className = "status-pill status-pill--ok";
    els.healthLabel.textContent = `Ready (${active})`;
  } catch {
    els.healthPill.className = "status-pill status-pill--warn";
    els.healthLabel.textContent = "API Offline";
  }

  // Load curated sample articles
  try {
    const data = await api.getSamples();
    store.samples = data.samples || [];
    store.samples.forEach((sample) => {
      const opt = document.createElement("option");
      opt.value = sample.id;
      opt.textContent = `[${sample.language.toUpperCase()}] ${sample.title}`;
      els.sampleSelect.appendChild(opt);
    });
  } catch {
    // Non-fatal if samples fail to load
  }

  // Event listeners
  if (els.heroSampleBtn) {
    els.heroSampleBtn.addEventListener("click", () => {
      if (store.samples.length > 0) {
        const preferred =
          store.samples.find((s) => s.language === store.uiLang) || store.samples[0];
        els.sampleSelect.value = preferred.id;
        els.articleInput.value = preferred.text;
        els.sourceLangSelect.value = "auto";
        clearInputError();
        updateClientStats();
        scheduleLanguageDetect();
      }
      if (els.studioWorkspace) {
        els.studioWorkspace.scrollIntoView({ behavior: "smooth" });
      }
    });
  }

  els.themeToggleBtn.addEventListener("click", () => {
    store.setTheme(store.theme === "dark" ? "light" : "dark");
    applyTheme();
  });

  els.uiLangBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      store.setUiLang(btn.dataset.uiLang);
      applyUiLanguage();
      updateClientStats();
    });
  });

  els.lengthBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      store.selectedLength = btn.dataset.length;
      els.lengthBtns.forEach((b) => b.classList.toggle("is-active", b === btn));
    });
  });

  els.sampleSelect.addEventListener("change", () => {
    const selected = store.samples.find((s) => s.id === els.sampleSelect.value);
    if (!selected) return;
    els.articleInput.value = selected.text;
    els.sourceLangSelect.value = "auto";
    clearInputError();
    updateClientStats();
    scheduleLanguageDetect();
  });

  els.fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    loadFileContent(file);
    e.target.value = "";
  });

  els.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropZone.classList.add("is-dragover");
  });
  els.dropZone.addEventListener("dragleave", () => {
    els.dropZone.classList.remove("is-dragover");
  });
  els.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropZone.classList.remove("is-dragover");
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    loadFileContent(file);
  });

  els.articleInput.addEventListener("input", () => {
    clearInputError();
    updateClientStats();
    scheduleLanguageDetect();
  });

  els.clearBtn.addEventListener("click", () => {
    els.articleInput.value = "";
    els.sampleSelect.value = "";
    clearInputError();
    updateClientStats();
    scheduleLanguageDetect();
  });

  els.summarizeBtn.addEventListener("click", handleSummarize);
  els.translateBtn.addEventListener("click", handleTranslate);
  els.retryTranslateBtn.addEventListener("click", handleTranslate);

  els.copySummaryBtn.addEventListener("click", async () => {
    if (!store.summaryResult) return;
    await navigator.clipboard.writeText(store.summaryResult.summary);
    showToast(t("copiedToast"));
  });

  els.copyTranslationBtn.addEventListener("click", async () => {
    if (!store.translationResult) return;
    await navigator.clipboard.writeText(store.translationResult.translated_text);
    showToast(t("copiedToast"));
  });

  els.exportMdBtn.addEventListener("click", () => {
    if (!store.summaryResult) return;
    const md = buildMarkdownReport({
      articleText: els.articleInput.value,
      summaryResult: store.summaryResult,
      translationResult: store.translationResult,
    });
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `news-summary-${store.summaryResult.detected_language}.md`;
    link.click();
    URL.revokeObjectURL(url);
  });

  els.historyToggleBtn.addEventListener("click", () => toggleHistoryDrawer());
  els.closeHistoryBtn.addEventListener("click", () => toggleHistoryDrawer(false));
  els.clearHistoryBtn.addEventListener("click", () => {
    store.clearHistory();
    renderHistory();
  });
}

init();
