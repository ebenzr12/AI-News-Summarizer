"""
AI News Summarizer - Configuration Module
=========================================
Centralized configuration management for paths, model parameters,
training hyperparameters, and dataset processing.
"""

from pathlib import Path
import os
import torch

# ==============================================================================
# 1. DIRECTORY & FILE PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Dataset paths
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

# Model & Checkpoint paths
MODELS_DIR = BASE_DIR / "models"
PRETRAINED_MODEL_DIR = MODELS_DIR / "pretrained"
FINE_TUNED_MODEL_DIR = MODELS_DIR / "fine_tuned"
TOKENIZER_DIR = MODELS_DIR / "tokenizer"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# Source & Config
SRC_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "config"

# ==============================================================================
# 2. MODEL & TASK CONFIGURATION
# ==============================================================================
MODEL_NAME = "google/mt5-small"

# Language-specific task prefixes
TASK_PREFIX_INDONESIAN = "summarize Indonesian: "
TASK_PREFIX_ENGLISH = "summarize English: "

LANGUAGE_PREFIX_MAP = {
    "id": TASK_PREFIX_INDONESIAN,
    "indonesian": TASK_PREFIX_INDONESIAN,
    "en": TASK_PREFIX_ENGLISH,
    "english": TASK_PREFIX_ENGLISH,
}

# ==============================================================================
# 3. TOKENIZATION & SEQUENCE LENGTHS
# ==============================================================================
# News articles are truncated at 512 tokens; summaries at 128 tokens
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

# ==============================================================================
# 4. DATASET SPECIFICATIONS (XL-Sum)
# ==============================================================================
DATASET_NAME = "csebuetnlp/xlsum"
SUPPORTED_LANGUAGES = ["indonesian", "english"]

# Established dataset splits
DATASET_SPLITS = {
    "indonesian": {"train": 38240, "validation": 4780, "test": 4780},
    "english": {"train": 306522, "validation": 11535, "test": 11535},
}

# Processed dataset file format
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train_bilingual.parquet"
PROCESSED_VAL_FILE = PROCESSED_DATA_DIR / "validation_bilingual.parquet"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test_bilingual.parquet"

# ==============================================================================
# 5. HARDWARE & TRAINING HYPERPARAMETERS (Optimized for 4GB VRAM)
# ==============================================================================
RANDOM_SEED = 42

# Hardware detection
CUDA_AVAILABLE = torch.cuda.is_available()
# RTX 3050 (Ampere) supports BF16, which prevents mT5 NaN loss
USE_BF16 = CUDA_AVAILABLE and torch.cuda.is_bf16_supported()
USE_FP16 = False  # DO NOT use FP16 for mT5 to prevent numerical instability/NaN

# Batch sizes & accumulation
PER_DEVICE_TRAIN_BATCH_SIZE = 2      # Keeps VRAM usage under 3.5GB
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16     # Effective batch size = 2 * 16 = 32

# Optimization
LEARNING_RATE = 5e-4                 # Standard learning rate for T5/Adafactor
OPTIMIZER_NAME = "adafactor"         # Memory-efficient optimizer for T5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

# Training limits & checkpoints
NUM_TRAIN_EPOCHS = 3
SAVE_TOTAL_LIMIT = 2
EVAL_STRATEGY = "steps"
EVAL_STEPS = 500
SAVE_STEPS = 500
LOGGING_STEPS = 100

# Generation settings for evaluation and inference
GEN_NUM_BEAMS = 4
GEN_MAX_LENGTH = 128
GEN_MIN_LENGTH = 20
GEN_LENGTH_PENALTY = 1.0
GEN_NO_REPEAT_NGRAM_SIZE = 2

# ==============================================================================
# 6. WEB APPLICATION, INFERENCE ENGINE & TRANSLATION SETTINGS
# ==============================================================================
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]

# Input validation limits
MIN_ARTICLE_WORDS = 10
MAX_ARTICLE_CHARS = 50000
MAX_TRANSLATE_CHARS = 15000

# Summarization engine selection: "auto" | "extractive" | "abstractive" | "llm"
DEFAULT_SUMMARY_ENGINE = os.getenv("SUMMARY_ENGINE", "auto")

# Summary length presets -> (min_sentences, max_sentences, ratio)
SUMMARY_LENGTH_PRESETS = {
    "concise": {"min_sentences": 1, "max_sentences": 2, "ratio": 0.22, "max_tokens": 80, "min_tokens": 16},
    "balanced": {"min_sentences": 2, "max_sentences": 4, "ratio": 0.32, "max_tokens": 128, "min_tokens": 24},
    "detailed": {"min_sentences": 3, "max_sentences": 6, "ratio": 0.45, "max_tokens": 192, "min_tokens": 40},
}

# Optional external OpenAI-compatible or Gemini API configuration (never hardcode secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Translation settings
TRANSLATION_MAX_CHUNK_CHARS = 450  # Safe margin below MyMemory 500-char limit
TRANSLATION_TIMEOUT_SECONDS = int(os.getenv("TRANSLATION_TIMEOUT_SECONDS", "12"))
TRANSLATION_CACHE_SIZE = 256
