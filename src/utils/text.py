"""
Text processing utilities.

Provides functions for:
- Stopword loading from NLTK
- Simple tokenization
"""

import nltk
from typing import Set, List, Optional


def load_stopwords() -> Set[str]:
    """
    Load English stopwords from NLTK.

    Downloads the stopwords corpus if not already present.

    Returns:
        Set of English stopwords
    """
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))


def simple_tokenizer(text: str, stopwords_set: Optional[Set[str]] = None) -> List[str]:
    """
    Simple whitespace tokenizer + lowercase + optional stopword removal.

    Args:
        text: Input text to tokenize
        stopwords_set: Optional set of stopwords to remove

    Returns:
        List of tokens
    """
    tokens = text.lower().split()
    if stopwords_set:
        return [t for t in tokens if t not in stopwords_set]
    return tokens


__all__ = ["load_stopwords", "simple_tokenizer"]
