"""Shared, conservative text normalization for training and inference."""
from __future__ import annotations

import html
import re
import unicodedata
from nltk.tokenize import TweetTokenizer

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
SPACE_RE = re.compile(r"\s+")
_TOKENIZER = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)


def preprocess_text(value: object) -> str:
    """Normalize a post while keeping negation, hashtags, punctuation, and emoji."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text).lower()
    text = URL_RE.sub(" url ", text)
    text = MENTION_RE.sub(" user ", text)
    # Keep Unicode letters, numbers, emoji/symbols, and expressive punctuation.
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if (char.isalnum() or char.isspace() or char in "#@'’!?.,:;-…" or
                category.startswith(("S", "M")) or char == "\u200d"):
            kept.append(char)
        else:
            kept.append(" ")
    text = "".join(kept)
    text = SPACE_RE.sub(" ", text).strip()
    return " ".join(_TOKENIZER.tokenize(text))
