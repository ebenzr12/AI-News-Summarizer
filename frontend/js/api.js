/**
 * HTTP API client with timeout control and structured error extraction.
 */

export class ApiClientError extends Error {
  constructor(message, { code = "REQUEST_FAILED", status = 0, requestId = null } = {}) {
    super(message);
    this.name = "ApiClientError";
    this.code = code;
    this.status = status;
    this.requestId = requestId;
  }
}

async function requestJson(path, { method = "GET", body = null, timeoutMs = 25000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }

    if (!response.ok) {
      const errObj = payload && payload.error ? payload.error : {};
      throw new ApiClientError(
        errObj.message || `Request failed with HTTP ${response.status}`,
        {
          code: errObj.code || "HTTP_ERROR",
          status: response.status,
          requestId: payload ? payload.request_id : null,
        }
      );
    }

    return payload;
  } catch (err) {
    if (err instanceof ApiClientError) {
      throw err;
    }
    if (err.name === "AbortError") {
      throw new ApiClientError("Request timed out. Please try again.", {
        code: "TIMEOUT",
        status: 408,
      });
    }
    throw new ApiClientError("Network error while contacting the backend API.", {
      code: "NETWORK_ERROR",
      status: 0,
    });
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  getHealth: () => requestJson("/api/v1/health", { timeoutMs: 8000 }),
  getSamples: () => requestJson("/api/v1/samples", { timeoutMs: 8000 }),
  detectLanguage: (text) =>
    requestJson("/api/v1/detect-language", { method: "POST", body: { text }, timeoutMs: 8000 }),
  summarize: ({ text, language, length, engine }) =>
    requestJson("/api/v1/summarize", {
      method: "POST",
      body: { text, language, length, engine },
      timeoutMs: 45000,
    }),
  translate: ({ text, sourceLanguage, targetLanguage }) =>
    requestJson("/api/v1/translate", {
      method: "POST",
      body: {
        text,
        source_language: sourceLanguage,
        target_language: targetLanguage,
      },
      timeoutMs: 25000,
    }),
};
