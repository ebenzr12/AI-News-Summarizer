# WartaAI — Bilingual AI News Summarizer (ID / EN)

Full-stack bilingual (Indonesian & English) AI News Summarization & Translation web application combining a modular Starlette/Pydantic REST API, responsive vanilla ES6+ SPA frontend, pluggable summarization engines (Fine-tuned `google/mt5-small`, optional Cloud LLM API, and deterministic TF-IDF + Lead-Bias + MMR extractive fallback), and a decoupled translation pipeline.

## Architecture Overview

```text
AI_News_Summarizer/
├── backend/
│   ├── app.py               # Starlette ASGI app, routes, middleware, static SPA hosting
│   ├── errors.py            # Structured domain -> HTTP JSON error handlers
│   ├── schemas.py           # Pydantic v2 request/response contracts
│   ├── services.py          # Application service facade & sample articles
│   └── server.py            # Uvicorn CLI runner
├── frontend/
│   ├── index.html           # Responsive editorial web UI (light/dark, EN/ID UI)
│   ├── css/                 # Design tokens, responsive CSS grid layout, components
│   └── js/                  # Modular ES6+ API client, state store, i18n, pure utils
├── src/
│   ├── extractive_summarizer.py # Bilingual TF-IDF + Lead-Bias + MMR summarizer
│   ├── summarizer.py        # Multi-engine summarization router (auto/extractive/abstractive/llm)
│   ├── translator.py        # Decoupled ID <-> EN translator with chunking & LRU cache
│   ├── language_detector.py # langdetect + ID/EN lexical disambiguation
│   ├── preprocessing.py     # Cleaning, abbreviation-safe sentence splitter, stats
│   ├── model_loader.py      # mT5-small loader, BF16 placement, checkpoint checks
│   ├── data_loader.py       # XL-Sum Parquet data pipeline
│   └── evaluation.py        # ROUGE-1/2/L/Lsum evaluation
├── config/
│   └── config.py            # Centralized settings & environment variables
├── tests/                   # Python unittest + Starlette TestClient + Node test suite
├── train.py                 # mT5-small Seq2Seq fine-tuning script
└── app.py                   # Legacy Streamlit interface
```

## Quick Start

### 1. Run the Full-Stack Web App (Recommended)
```bash
.venv\Scripts\python.exe -m backend.server
```
Open `http://127.0.0.1:8000` in your browser.

### 2. Optional Environment Variables
- `API_HOST` / `API_PORT` (default `127.0.0.1:8000`)
- `GEMINI_API_KEY` or `OPENAI_API_KEY` (optional cloud LLM summarization & translation)
- `SUMMARY_ENGINE` (`auto`, `extractive`, `abstractive`, `llm`)

### 3. Fine-Tune `google/mt5-small` (Optional Abstractive Weights)
```bash
.venv\Scripts\python.exe train.py --sample_id 1000 --sample_en 1000 --epochs 3
```
Once `models/fine_tuned/` contains the checkpoint, `engine="auto"` automatically upgrades to the fine-tuned mT5 Seq2Seq model.

### 4. Run Automated Tests
```bash
.venv\Scripts\python.exe -m unittest discover -s tests -v
node --test tests/frontend_utils.test.mjs
```

### 5. Legacy Streamlit App
```bash
.venv\Scripts\streamlit.exe run app.py
```

## REST API Endpoints
- `GET /api/v1/health` — Engine readiness, CUDA availability, supported languages
- `GET /api/v1/samples` — Curated Indonesian & English sample news articles
- `POST /api/v1/detect-language` — Language detection (`id` / `en`) & text statistics
- `POST /api/v1/summarize` — Generate concise summary in original language
- `POST /api/v1/translate` — Decoupled summary translation (`id <-> en`)
