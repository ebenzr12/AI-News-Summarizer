"""
AI News Summarizer - Data Loader Module
=======================================
Loads, cleans, combines, and persists the bilingual XL-Sum dataset
(Indonesian & English) with zero data leakage and strict reproducibility.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_TRAIN_FILE,
    PROCESSED_VAL_FILE,
    PROCESSED_TEST_FILE,
    SUPPORTED_LANGUAGES,
    RANDOM_SEED,
)
from src.preprocessing import clean_news_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def are_processed_datasets_available() -> bool:
    """
    Check whether processed dataset files already exist locally.

    Returns:
        bool: True if train, validation, and test parquet files all exist.
    """
    exists = (
        PROCESSED_TRAIN_FILE.exists()
        and PROCESSED_VAL_FILE.exists()
        and PROCESSED_TEST_FILE.exists()
    )
    if exists:
        logger.info(
            f"Processed datasets found locally:\n"
            f"  - Train: {PROCESSED_TRAIN_FILE}\n"
            f"  - Val:   {PROCESSED_VAL_FILE}\n"
            f"  - Test:  {PROCESSED_TEST_FILE}"
        )
    return exists


def load_processed_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load pre-processed bilingual datasets from local Parquet files.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with 'train', 'validation', and 'test' DataFrames.
    """
    if not are_processed_datasets_available():
        raise FileNotFoundError(
            "Processed dataset files not found. Run download_and_process_xlsum() first."
        )

    logger.info("Loading processed datasets from local Parquet files...")
    return {
        "train": pd.read_parquet(PROCESSED_TRAIN_FILE),
        "validation": pd.read_parquet(PROCESSED_VAL_FILE),
        "test": pd.read_parquet(PROCESSED_TEST_FILE),
    }


def clean_dataframe(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Clean dataset rows:
    - Drop null/NaN values
    - Remove empty text or summary strings
    - Normalize whitespace
    - Add language identifier column

    Args:
        df (pd.DataFrame): DataFrame with 'text' and 'summary' columns.
        language (str): Language identifier ('indonesian' or 'english').

    Returns:
        pd.DataFrame: Cleaned DataFrame with ['id', 'text', 'summary', 'language'].
    """
    initial_len = len(df)

    # Standardize column names if needed
    rename_cols = {}
    if "maintext" in df.columns and "text" not in df.columns:
        rename_cols["maintext"] = "text"
    if rename_cols:
        df = df.rename(columns=rename_cols)

    # Drop nulls
    df = df.dropna(subset=["text", "summary"]).copy()

    # Clean text content
    df["text"] = df["text"].astype(str).apply(clean_news_text)
    df["summary"] = df["summary"].astype(str).apply(clean_news_text)

    # Filter out empty or trivially short entries (< 3 words)
    valid_mask = (
        (df["text"].str.split().str.len() >= 5)
        & (df["summary"].str.split().str.len() >= 2)
    )
    df = df[valid_mask].copy()

    # Add language tag
    df["language"] = language

    # Reset index and ensure unique identifier
    df = df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = [f"{language}_{i}" for i in range(len(df))]

    # Keep only relevant columns
    df = df[["id", "text", "summary", "language"]]

    logger.info(
        f"[{language.upper()}] Cleaned: {len(df)} valid samples retained from {initial_len} initial."
    )
    return df


def download_and_process_xlsum(
    languages: Optional[list] = None,
    save_local: bool = True,
    sample_sizes: Optional[Dict[str, int]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Download raw XL-Sum dataset for specified languages, clean, combine,
    shuffle reproducibly, and optionally save locally to Parquet format.

    Args:
        languages (list, optional): Languages to process (default: ['indonesian', 'english']).
        save_local (bool): Whether to persist processed files to disk.
        sample_sizes (dict, optional): Optional sample limit per language for testing/PoC.

    Returns:
        Dict[str, pd.DataFrame]: Processed train, validation, and test DataFrames.
    """
    from datasets import load_dataset

    if languages is None:
        languages = SUPPORTED_LANGUAGES

    # Check if already processed locally
    if save_local and are_processed_datasets_available() and sample_sizes is None:
        logger.info("Local processed datasets already exist. Skipping download.")
        return load_processed_datasets()

    splits = ["train", "validation", "test"]
    combined_data = {split: [] for split in splits}

    for lang in languages:
        logger.info(f"Fetching XL-Sum dataset for language: {lang}...")
        for split in splits:
            # Download from Hugging Face
            ds = load_dataset(
                "csebuetnlp/xlsum",
                lang,
                split=split,
                cache_dir=str(RAW_DATA_DIR),
            )
            df = ds.to_pandas()

            # Clean and add language label
            cleaned_df = clean_dataframe(df, language=lang)

            # If sample_sizes provided (e.g. for quick local training / PoC)
            if sample_sizes and lang in sample_sizes and split == "train":
                target_n = min(sample_sizes[lang], len(cleaned_df))
                cleaned_df = cleaned_df.sample(n=target_n, random_state=RANDOM_SEED).reset_index(drop=True)
                logger.info(f"Sampled {target_n} train samples for {lang}.")

            combined_data[split].append(cleaned_df)

    final_datasets = {}
    for split in splits:
        # Concatenate Indonesian and English datasets
        combined_df = pd.concat(combined_data[split], ignore_index=True)

        # Reproducibly shuffle (especially for training split)
        if split == "train":
            combined_df = combined_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        final_datasets[split] = combined_df
        logger.info(f"Final combined '{split}' set: {len(combined_df)} samples.")

    # Save to disk
    if save_local:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        final_datasets["train"].to_parquet(PROCESSED_TRAIN_FILE, index=False)
        final_datasets["validation"].to_parquet(PROCESSED_VAL_FILE, index=False)
        final_datasets["test"].to_parquet(PROCESSED_TEST_FILE, index=False)
        logger.info(f"Saved processed datasets to: {PROCESSED_DATA_DIR}")

    return final_datasets
