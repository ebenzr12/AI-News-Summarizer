"""
AI News Summarizer - Model Loader Module
========================================
Manages loading, caching, device placement, and persistence of the
Google mT5-small model and tokenizer.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    MODEL_NAME,
    PRETRAINED_MODEL_DIR,
    FINE_TUNED_MODEL_DIR,
    TOKENIZER_DIR,
    USE_BF16,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_fine_tuned_model_available() -> bool:
    """
    Check whether a locally fine-tuned Seq2Seq model checkpoint exists.
    """
    return (
        FINE_TUNED_MODEL_DIR.exists()
        and (FINE_TUNED_MODEL_DIR / "config.json").exists()
    )


def get_device() -> torch.device:
    """
    Detect and return the appropriate execution device (CUDA or CPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU device: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA is not available. Using CPU fallback (slow for training).")
    return device


def get_torch_dtype() -> torch.dtype:
    """
    Select optimal torch dtype based on GPU capability.
    Uses bfloat16 on Ampere+ (RTX 3050) to prevent mT5 NaN instability.
    """
    if USE_BF16:
        logger.info("Using bfloat16 precision for numerical stability and memory efficiency.")
        return torch.bfloat16
    return torch.float32


def load_tokenizer(cache_dir: Optional[Path] = TOKENIZER_DIR) -> AutoTokenizer:
    """
    Load mT5-small tokenizer. Checks local directory first before downloading.

    Args:
        cache_dir (Path, optional): Directory where tokenizer is saved locally.

    Returns:
        AutoTokenizer: Initialized tokenizer.
    """
    cache_path = Path(cache_dir)

    # Check if tokenizer already exists locally
    if cache_path.exists() and (cache_path / "tokenizer_config.json").exists():
        logger.info(f"Loading tokenizer from local cache: {cache_path}")
        return AutoTokenizer.from_pretrained(str(cache_path))

    logger.info(f"Downloading tokenizer '{MODEL_NAME}' from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Save locally to avoid repeated downloads
    cache_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(cache_path))
    logger.info(f"Tokenizer saved locally to: {cache_path}")

    return tokenizer


def load_model(
    model_type: str = "auto",
    device: Optional[torch.device] = None,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    Load mT5-small model and tokenizer.
    Checks fine-tuned first if 'auto' or 'fine_tuned' is specified.
    Never overwrites the pretrained baseline model.

    Args:
        model_type (str): 'auto', 'fine_tuned', or 'pretrained'.
        device (torch.device, optional): Target hardware device.

    Returns:
        Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]: Model and Tokenizer instances.
    """
    if device is None:
        device = get_device()

    dtype = get_torch_dtype()
    tokenizer = load_tokenizer()

    # Determine load source
    model_path = None
    has_fine_tuned = (
        FINE_TUNED_MODEL_DIR.exists()
        and (FINE_TUNED_MODEL_DIR / "config.json").exists()
    )

    if model_type in ("auto", "fine_tuned") and has_fine_tuned:
        logger.info(f"Loading FINE-TUNED model from: {FINE_TUNED_MODEL_DIR}")
        model_path = str(FINE_TUNED_MODEL_DIR)
    else:
        # Check if pretrained model exists locally
        if PRETRAINED_MODEL_DIR.exists() and (PRETRAINED_MODEL_DIR / "config.json").exists():
            logger.info(f"Loading PRETRAINED model from local cache: {PRETRAINED_MODEL_DIR}")
            model_path = str(PRETRAINED_MODEL_DIR)
        else:
            logger.info(f"Downloading base PRETRAINED model '{MODEL_NAME}' from Hugging Face...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=dtype,
            )
            # Save pretrained baseline separately so it is never overwritten
            PRETRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(PRETRAINED_MODEL_DIR))
            logger.info(f"Base pretrained model saved locally to: {PRETRAINED_MODEL_DIR}")
            model = model.to(device)
            return model, tokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()

    return model, tokenizer
