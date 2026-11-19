"""
src/preprocessing.py
Comprehensive Text Preprocessing Pipeline for TopicLens AI.
Handles lowercase conversion, HTML/URL removal, punctuation stripping, number handling,
tokenization, stopword removal, lemmatization, duplicate detection, and before-and-after diagnostics.
"""

import re
import string
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd


# Default standard English stopwords fallback (covers NLTK standard 179 + typical noise words)
DEFAULT_STOPWORDS: Set[str] = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't",
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
    'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',
    'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how',
    "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its',
    'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on',
    'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't",
    'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's",
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd",
    "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
    'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when',
    "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't",
    'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves',
    # Common conversational / corpus fillers
    'also', 'get', 'like', 'one', 'would', 'use', 'make', 'may', 'well', 'said', 'say', 'go', 'know', 'see'
}


class TextPreprocessor:
    """
    Configurable NLP Preprocessor for document collections.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_word_len: int = 3,
        custom_stopwords: Optional[List[str]] = None
    ):
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_len = min_word_len

        # Initialize Stopwords
        self.stopwords = set(DEFAULT_STOPWORDS)
        self._load_nltk_stopwords()
        if custom_stopwords:
            self.stopwords.update([w.strip().lower() for w in custom_stopwords if w.strip()])

        # Initialize Lemmatizer
        self.lemmatizer = None
        if self.lemmatize:
            self._load_lemmatizer()

    def _load_nltk_stopwords(self) -> None:
        """Attempt to load NLTK stopwords, fallback to DEFAULT_STOPWORDS if not available."""
        try:
            import nltk
            try:
                from nltk.corpus import stopwords
                nltk_stops = set(stopwords.words('english'))
                self.stopwords.update(nltk_stops)
            except LookupError:
                nltk.download('stopwords', quiet=True)
                from nltk.corpus import stopwords
                self.stopwords.update(stopwords.words('english'))
        except Exception:
            pass

    def _load_lemmatizer(self) -> None:
        """Attempt to initialize WordNetLemmatizer with fallback."""
        try:
            import nltk
            try:
                from nltk.stem import WordNetLemmatizer
                lem = WordNetLemmatizer()
                # Test lemmatizer
                lem.lemmatize('testing')
                self.lemmatizer = lem
            except LookupError:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                from nltk.stem import WordNetLemmatizer
                self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None

    def clean_text(self, text: str) -> str:
        """
        Execute text cleaning steps based on preprocessor configuration.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Lowercase conversion
        if self.lowercase:
            text = text.lower()

        # 2. HTML tag removal
        if self.remove_html:
            text = re.sub(r'<[^>]+>', ' ', text)

        # 3. URL removal
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

        # 4. Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # 5. Number handling
        if self.remove_numbers:
            text = re.sub(r'\b\d+\b', ' ', text)
            # Remove isolated digits and alphanumeric hashes if desired
            text = re.sub(r'\d+', ' ', text)

        # 6. Punctuation cleaning
        if self.remove_punctuation:
            # Replace punctuation with whitespace to prevent word concatenation
            text = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', text)

        # 7. Tokenization via regex
        tokens = re.findall(r'\b[a-zA-Z]{' + str(self.min_word_len) + r',}\b', text)

        # 8. Stopword removal
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        # 9. Lemmatization
        if self.lemmatize and self.lemmatizer:
            try:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            except Exception:
                pass

        # Re-filter tokens that might have become too short or stopwords after lemmatization
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords and len(t) >= self.min_word_len]

        return " ".join(tokens)


def preprocess_corpus(
    df: pd.DataFrame,
    text_column: str,
    preprocessor: Optional[TextPreprocessor] = None,
    drop_empty: bool = True,
    drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply preprocessing to the entire dataframe text column.
    
    Returns:
        processed_df: DataFrame with added columns:
            - 'clean_text': preprocessed string
            - 'tokens_list': list of tokens
            - 'clean_word_count': number of cleaned words
        stats: Dictionary of preprocessing metrics & diagnostics
    """
    if preprocessor is None:
        preprocessor = TextPreprocessor()

    original_count = len(df)
    work_df = df.copy()

    # Ensure text is string and fill NaNs
    work_df[text_column] = work_df[text_column].fillna("").astype(str)

    # Detect duplicates in raw text
    raw_duplicates_count = int(work_df[text_column].duplicated().sum())

    # Apply cleaning
    work_df['clean_text'] = work_df[text_column].apply(preprocessor.clean_text)
    work_df['tokens_list'] = work_df['clean_text'].apply(lambda x: x.split() if x else [])
    work_df['clean_word_count'] = work_df['tokens_list'].apply(len)

    # Identify empty documents after cleaning
    empty_after_clean_count = int((work_df['clean_word_count'] == 0).sum())

    # Detect duplicates in cleaned text
    clean_duplicates_count = int(work_df['clean_text'].duplicated().sum())

    # Filter empty documents if requested
    if drop_empty:
        work_df = work_df[work_df['clean_word_count'] > 0].copy()

    # Filter duplicates if requested
    if drop_duplicates:
        work_df = work_df.drop_duplicates(subset=['clean_text']).copy()

    work_df.reset_index(drop=True, inplace=True)
    final_count = len(work_df)

    # Compute vocabulary metrics
    all_raw_tokens = []
    for raw_t in df[text_column].dropna():
        all_raw_tokens.extend(raw_t.lower().split())

    all_clean_tokens = [tok for tokens in work_df['tokens_list'] for tok in tokens]

    raw_vocab_size = len(set(all_raw_tokens))
    clean_vocab_size = len(set(all_clean_tokens))

    stats = {
        'original_document_count': original_count,
        'final_document_count': final_count,
        'removed_empty_count': empty_after_clean_count,
        'raw_duplicates_count': raw_duplicates_count,
        'clean_duplicates_count': clean_duplicates_count,
        'raw_total_tokens': len(all_raw_tokens),
        'clean_total_tokens': len(all_clean_tokens),
        'raw_vocab_size': raw_vocab_size,
        'clean_vocab_size': clean_vocab_size,
        'vocab_reduction_percent': float(round((1 - (clean_vocab_size / max(raw_vocab_size, 1))) * 100, 2)),
        'avg_clean_doc_length': float(round(work_df['clean_word_count'].mean(), 2)) if final_count > 0 else 0.0
    }

    return work_df, stats


def get_before_after_samples(
    original_series: pd.Series,
    cleaned_series: pd.Series,
    n_samples: int = 3
) -> List[Dict[str, str]]:
    """
    Extract paired before-and-after samples for visual comparison in the dashboard.
    """
    samples = []
    indices = original_series.index[:n_samples]
    for idx in indices:
        orig = str(original_series.loc[idx])
        clean = str(cleaned_series.loc[idx])
        samples.append({
            'doc_index': idx,
            'original': orig[:350] + ("..." if len(orig) > 350 else ""),
            'cleaned': clean[:350] + ("..." if len(clean) > 350 else "")
        })
    return samples
