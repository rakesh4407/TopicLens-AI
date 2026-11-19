"""
src/data_loader.py
Dataset loading, parsing, validation, and statistical summary utilities for TopicLens AI.
Supports CSV, multiple TXT files, and built-in sample datasets (20 Newsgroups, Customer Feedback, Tech News).
"""

from io import StringIO, BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def detect_text_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect columns in a DataFrame that likely contain natural language text.
    Prioritizes columns with object/string dtype, high uniqueness, and longer average string length.
    """
    text_cols = []
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0:
                # Calculate mean character length of strings
                sample = non_null.astype(str).head(100)
                mean_len = sample.str.len().mean()
                if mean_len >= 15:  # likely sentence or document
                    text_cols.append(col)
    
    # Fallback: if no column met threshold, return all object/string columns
    if not text_cols:
        text_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
        
    return text_cols if text_cols else list(df.columns)


def load_csv_data(
    file_or_path: Union[str, Path, BytesIO, StringIO, Any],
    text_column: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Load a CSV file with robust encoding detection.
    
    Returns:
        df: pd.DataFrame with the loaded data
        text_cols: list of potential text columns
        selected_text_col: automatically chosen or confirmed text column
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig', 'iso-8859-1']
    df = None
    last_err = None

    for enc in encodings:
        try:
            if hasattr(file_or_path, 'seek'):
                file_or_path.seek(0)
            df = pd.read_csv(file_or_path, encoding=enc)
            break
        except Exception as e:
            last_err = e
            continue

    if df is None:
        raise ValueError(f"Failed to read CSV with supported encodings. Last error: {last_err}")

    text_cols = detect_text_columns(df)
    
    if text_column and text_column in df.columns:
        selected_text_col = text_column
    elif text_cols:
        selected_text_col = text_cols[0]
    else:
        selected_text_col = df.columns[0]

    return df, text_cols, selected_text_col


def load_txt_files(uploaded_files: List[Any]) -> pd.DataFrame:
    """
    Load multiple TXT files from uploaded file objects or file paths.
    
    Returns:
        pd.DataFrame with columns: ['doc_id', 'filename', 'text', 'char_count', 'word_count']
    """
    records = []
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig']

    for idx, f in enumerate(uploaded_files, start=1):
        filename = getattr(f, 'name', f"document_{idx}.txt")
        raw_text = ""

        if hasattr(f, 'read'):
            raw_bytes = f.read()
            if isinstance(raw_bytes, str):
                raw_text = raw_bytes
            else:
                for enc in encodings:
                    try:
                        raw_text = raw_bytes.decode(enc)
                        break
                    except (UnicodeDecodeError, AttributeError):
                        continue
        else:
            # Assume file path
            path = Path(f)
            filename = path.name
            for enc in encodings:
                try:
                    with open(path, 'r', encoding=enc) as fh:
                        raw_text = fh.read()
                    break
                except UnicodeDecodeError:
                    continue

        words = raw_text.split()
        records.append({
            'doc_id': idx,
            'filename': filename,
            'text': raw_text,
            'char_count': len(raw_text),
            'word_count': len(words)
        })

    return pd.DataFrame(records)


def get_dataset_summary(df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
    """
    Compute comprehensive exploratory data statistics on the selected text column.
    """
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in dataframe.")

    series = df[text_column]
    total_rows = len(df)
    null_count = int(series.isna().sum())
    
    # Filter non-null strings
    non_null_series = series.dropna().astype(str)
    empty_strings = int((non_null_series.str.strip() == '').sum())
    valid_series = non_null_series[non_null_series.str.strip() != '']
    valid_count = len(valid_series)
    
    # Duplicates detection
    duplicate_count = int(valid_series.duplicated().sum())

    # Word and character statistics
    word_counts = valid_series.apply(lambda x: len(x.split()))
    char_counts = valid_series.apply(len)

    stats = {
        'total_documents': total_rows,
        'valid_documents': valid_count,
        'missing_values': null_count,
        'empty_documents': empty_strings,
        'duplicate_documents': duplicate_count,
        'text_column': text_column,
        'columns': list(df.columns),
        'avg_word_count': float(round(word_counts.mean(), 2)) if valid_count > 0 else 0.0,
        'median_word_count': float(round(word_counts.median(), 2)) if valid_count > 0 else 0.0,
        'min_word_count': int(word_counts.min()) if valid_count > 0 else 0,
        'max_word_count': int(word_counts.max()) if valid_count > 0 else 0,
        'avg_char_count': float(round(char_counts.mean(), 2)) if valid_count > 0 else 0.0,
        'total_word_count': int(word_counts.sum()) if valid_count > 0 else 0
    }
    return stats


def load_sample_dataset(dataset_type: str = '20newsgroups') -> Tuple[pd.DataFrame, str]:
    """
    Load curated sample datasets for instant demonstration and testing.
    Uses the SAMPLE_DATASETS registry from src.config for path/column resolution.
    
    dataset_type options:
        '20newsgroups'   — 20 Newsgroups subset (Tech, Space, Autos, Politics)
        'customer_feedback' — Synthetic customer feedback reviews
        'tech_news'      — Synthetic technology news articles
    """
    try:
        from src.config import SAMPLE_DATASETS, SAMPLE_DIR
        sample_dir = SAMPLE_DIR
    except ImportError:
        sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
        SAMPLE_DATASETS = []

    sample_dir.mkdir(parents=True, exist_ok=True)

    # Look up dataset in registry
    registry = {ds["id"]: ds for ds in SAMPLE_DATASETS} if SAMPLE_DATASETS else {}
    
    if dataset_type in registry:
        ds_info = registry[dataset_type]
        file_path = sample_dir / ds_info["file"]
        text_col = ds_info["text_col"]
        if file_path.exists():
            df = pd.read_csv(file_path)
            return df, text_col
    
    # Legacy fallback: match old dataset_type strings
    legacy_map = {
        '20newsgroups_tech': ('20newsgroups_subset.csv', 'text'),
        'customer_feedback': ('customer_feedback.csv', 'feedback_text'),
        'tech_news': ('tech_news.csv', 'content'),
    }
    if dataset_type in legacy_map:
        fname, col = legacy_map[dataset_type]
        file_path = sample_dir / fname
        if file_path.exists():
            df = pd.read_csv(file_path)
            return df, col

    # Default: 20 Newsgroups subset
    file_path = sample_dir / "20newsgroups_subset.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        return df, 'text'

    # Fallback: fetch from sklearn
    try:
        from sklearn.datasets import fetch_20newsgroups
        categories = ['comp.graphics', 'sci.space', 'rec.autos', 'talk.politics.guns']
        raw = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
        df = pd.DataFrame({
            'doc_id': range(1, len(raw.data) + 1),
            'text': raw.data,
            'original_category': [raw.target_names[t] for t in raw.target]
        })
        df = df[df['text'].str.strip().str.len() > 30].reset_index(drop=True)
        df.to_csv(file_path, index=False)
        return df, 'text'
    except Exception:
        return _generate_fallback_dataset(), 'text'


def _generate_fallback_dataset() -> pd.DataFrame:
    """Generate a clean curated dataset of multi-topic documents if offline."""
    tech_docs = [
        "Artificial intelligence and deep neural networks are transforming computer vision and automated diagnostics.",
        "Cloud computing architectures with microservices provide scalable infrastructure for enterprise web platforms.",
        "Quantum computing algorithms utilize qubit superposition to solve complex optimization problems exponentially faster.",
        "Database indexing with B-trees and hash indexes accelerates query execution time in transactional relational databases.",
        "Cybersecurity frameworks must protect cloud storage from ransomware, phishing vectors, and zero-day vulnerabilities."
    ] * 12

    space_docs = [
        "NASA James Webb Space Telescope detected ancient galactic formations near the observable cosmic cosmological boundary.",
        "Mars rover missions analyze regolith mineral samples to detect biosignatures and subterranean water ice formations.",
        "Orbital mechanics dictate spacecraft trajectory calculations for gravitational slingshot maneuvers past Jupiter.",
        "Astronomy researchers study exoplanet atmospheric spectral lines to evaluate habitable zone chemical compositions.",
        "Rocket propulsion systems utilizing cryogenic liquid methane and oxygen improve specific impulse and engine reusability."
    ] * 12

    health_docs = [
        "Cardiovascular exercise and balanced dietary nutrition lower arterial blood pressure and metabolic syndrome risks.",
        "Clinical trials evaluate mRNA vaccine efficacy and synthetic antigen antibody responses across diverse patient cohorts.",
        "Neuroscience research reveals neuroplasticity mechanisms during sleep consolidation and cognitive memory retention.",
        "Pharmacological therapies targeting cellular inflammation demonstrate promising outcomes in autoimmune disorder treatments.",
        "Epidemiological models simulate infectious disease transmission rates to guide public health containment interventions."
    ] * 12

    finance_docs = [
        "Central bank interest rate decisions and monetary policies directly influence sovereign bond yield curve dynamics.",
        "Portfolio diversification across global equity markets and fixed income securities mitigates systematic investment risk.",
        "Fintech innovations in distributed ledger settlements and automated clearinghouses lower cross-border transaction fees.",
        "Algorithmic high-frequency trading models leverage statistical arbitrage to exploit transient market liquidity disparities.",
        "Corporate earnings reports and quarterly revenue balance sheets indicate strong consumer spending resilience."
    ] * 12

    all_docs = tech_docs + space_docs + health_docs + finance_docs
    labels = (['Technology & AI'] * len(tech_docs) +
              ['Space Exploration'] * len(space_docs) +
              ['Healthcare & Medicine'] * len(health_docs) +
              ['Finance & Economics'] * len(finance_docs))
    
    df = pd.DataFrame({
        'doc_id': range(1, len(all_docs) + 1),
        'category': labels,
        'text': all_docs
    })
    return df
