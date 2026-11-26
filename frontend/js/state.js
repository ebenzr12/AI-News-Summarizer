/**
 * Reactive client-side state store with localStorage persistence.
 */

const STORAGE_KEYS = {
  THEME: "wartaai.theme",
  UI_LANG: "wartaai.uiLang",
  HISTORY: "wartaai.history",
};

const MAX_HISTORY = 10;

function readStorage(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw !== null ? raw : fallback;
  } catch {
    return fallback;
  }
}

function readJsonStorage(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function writeStorage(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Ignore storage quota / private-mode restrictions
  }
}

export const store = {
  theme: readStorage(STORAGE_KEYS.THEME, "light"),
  uiLang: readStorage(STORAGE_KEYS.UI_LANG, "en"),
  selectedLength: "balanced",
  summaryResult: null,
  translationResult: null,
  samples: [],
  history: readJsonStorage(STORAGE_KEYS.HISTORY, []),

  setTheme(theme) {
    this.theme = theme === "dark" ? "dark" : "light";
    writeStorage(STORAGE_KEYS.THEME, this.theme);
  },

  setUiLang(lang) {
    this.uiLang = lang === "id" ? "id" : "en";
    writeStorage(STORAGE_KEYS.UI_LANG, this.uiLang);
  },

  addHistory(entry) {
    const record = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2, 6)}`,
      createdAt: new Date().toISOString(),
      ...entry,
    };
    this.history = [record, ...this.history].slice(0, MAX_HISTORY);
    writeStorage(STORAGE_KEYS.HISTORY, JSON.stringify(this.history));
  },

  clearHistory() {
    this.history = [];
    writeStorage(STORAGE_KEYS.HISTORY, "[]");
  },
};
