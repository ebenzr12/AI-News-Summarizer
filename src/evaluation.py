"""
AI News Summarizer - Evaluation Module
======================================
Evaluates the summarization model using ROUGE metrics (ROUGE-1, ROUGE-2,
ROUGE-L, ROUGE-Lsum) separately on Indonesian and English test sets.
Generates structured qualitative sample comparisons.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    GEN_NUM_BEAMS,
    GEN_MAX_LENGTH,
    GEN_MIN_LENGTH,
    GEN_LENGTH_PENALTY,
)
from src.summarizer import NewsSummarizer
from src.preprocessing import clean_news_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rouge_scores(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute standard ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum metrics (F1 scores).
    Uses google's rouge_score library directly (fully offline & fast).

    Args:
        predictions (List[str]): List of generated summary strings.
        references (List[str]): List of ground-truth reference summary strings.

    Returns:
        Dict[str, float]: Scores rounded to 2 decimal places (scaled to 0-100%).
    """
    if len(predictions) != len(references):
        raise ValueError("Jumlah prediksi dan referensi harus sama.")

    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )

    r1_list = []
    r2_list = []
    rl_list = []
    rlsum_list = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rl_list.append(scores["rougeL"].fmeasure)
        rlsum_list.append(scores["rougeLsum"].fmeasure)

    return {
        "rouge1": round(float(np.mean(r1_list)) * 100, 2),
        "rouge2": round(float(np.mean(r2_list)) * 100, 2),
        "rougeL": round(float(np.mean(rl_list)) * 100, 2),
        "rougeLsum": round(float(np.mean(rlsum_list)) * 100, 2),
    }


def evaluate_test_dataset(
    summarizer: NewsSummarizer,
    test_df: pd.DataFrame,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate test samples separately for Indonesian and English.

    Args:
        summarizer (NewsSummarizer): Initialized summarizer instance.
        test_df (pd.DataFrame): Test DataFrame containing 'text', 'summary', 'language'.
        max_samples (int, optional): Max samples per language to evaluate (useful for fast testing).

    Returns:
        Dict[str, Any]: Evaluation report including separate metrics for Indonesian & English,
                        overall combined metrics, and qualitative sample comparisons.
    """
    report = {
        "languages": {},
        "sample_comparisons": [],
    }

    languages = test_df["language"].unique().tolist()

    all_preds = []
    all_refs = []

    for lang in languages:
        lang_df = test_df[test_df["language"] == lang]
        if max_samples:
            lang_df = lang_df.head(max_samples)

        logger.info(f"Evaluating {len(lang_df)} samples for language: {lang}...")

        preds = []
        refs = []
        sample_records = []

        for idx, row in lang_df.iterrows():
            article = row["text"]
            reference = row["summary"]

            # Generate summary
            res = summarizer.summarize(article)
            generated = res["summary"]

            preds.append(generated)
            refs.append(reference)

            # Record detailed comparison for sample inspection
            sample_records.append({
                "language": lang,
                "original_article": article,
                "reference_summary": reference,
                "generated_summary": generated,
            })

        # Calculate metrics for this language
        lang_scores = compute_rouge_scores(preds, refs)
        report["languages"][lang] = {
            "num_samples": len(lang_df),
            "metrics": lang_scores,
        }

        # Keep up to 3 comparison samples per language for qualitative review
        report["sample_comparisons"].extend(sample_records[:3])

        all_preds.extend(preds)
        all_refs.extend(refs)

    # Calculate overall combined metrics
    if all_preds:
        overall_scores = compute_rouge_scores(all_preds, all_refs)
        report["overall"] = {
            "total_samples": len(all_preds),
            "metrics": overall_scores,
        }

    return report


def print_evaluation_summary(report: Dict[str, Any]):
    """
    Format and print evaluation metrics report clearly.
    """
    print("\n" + "=" * 60)
    print("           MODEL EVALUATION REPORT (ROUGE SCORES)")
    print("=" * 60)

    for lang, data in report.get("languages", {}).items():
        print(f"\nLanguage: {lang.upper()} (Evaluated samples: {data['num_samples']})")
        m = data["metrics"]
        print(f"  - ROUGE-1   : {m['rouge1']:.2f}%")
        print(f"  - ROUGE-2   : {m['rouge2']:.2f}%")
        print(f"  - ROUGE-L   : {m['rougeL']:.2f}%")
        print(f"  - ROUGE-Lsum: {m['rougeLsum']:.2f}%")

    if "overall" in report:
        print(f"\nOVERALL COMBINED ({report['overall']['total_samples']} samples):")
        m = report["overall"]["metrics"]
        print(f"  - ROUGE-1   : {m['rouge1']:.2f}%")
        print(f"  - ROUGE-2   : {m['rouge2']:.2f}%")
        print(f"  - ROUGE-L   : {m['rougeL']:.2f}%")
        print(f"  - ROUGE-Lsum: {m['rougeLsum']:.2f}%")

    print("\n" + "=" * 60)
