"""Framework-independent caption preprocessing."""

import re
import string


class CaptionCleaner:
    """Normalize captions and add sequence boundary tokens."""

    def __init__(self, start_token: str = "startseq", end_token: str = "endseq") -> None:
        self.start_token = start_token
        self.end_token = end_token

    def clean(self, caption: str) -> str:
        normalized = caption.lower()
        normalized = normalized.translate(str.maketrans("", "", string.punctuation))
        normalized = re.sub(r"\s+", " ", normalized).strip()
        words = [word for word in normalized.split() if len(word) > 1]
        return " ".join([self.start_token, *words, self.end_token])

    def clean_mapping(self, mapping: dict[str, list[str]]) -> dict[str, list[str]]:
        return {image_id: [self.clean(caption) for caption in captions] for image_id, captions in mapping.items()}


class SimpleTokenizer:
    """Small tokenizer used by evaluation and tests without a framework dependency."""

    def tokenize(self, text: str) -> list[str]:
        return CaptionCleaner().clean(text).split()