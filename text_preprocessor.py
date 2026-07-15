"""
============================================================
TEXT PREPROCESSOR
Epic 3 - Story 1
============================================================
"""

import re
import nltk

# Download tokenizer once
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from nltk.tokenize import word_tokenize


class TextPreprocessor:

    def __init__(self):

        self.stop_words = {
            "the",
            "a",
            "an"
        }

    # -------------------------------------------------------
    # Clean text
    # -------------------------------------------------------

    def clean_text(self, text):

        if text is None:
            return ""

        text = str(text)

        # Lowercase
        text = text.lower()

        # Preserve ! and ? because they may indicate emotion
        text = re.sub(
            r"[^a-zA-Z0-9\s!?]",
            " ",
            text
        )

        # Remove extra spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove only a few basic stop words
        cleaned_tokens = [

            token

            for token in tokens

            if token not in self.stop_words
            and len(token) > 1

        ]

        return " ".join(cleaned_tokens)

    # -------------------------------------------------------
    # Tokenize only
    # -------------------------------------------------------

    def tokenize(self, text):

        cleaned = self.clean_text(text)

        return cleaned.split()

    # -------------------------------------------------------
    # Preprocess
    # -------------------------------------------------------

    def preprocess(self, text):

        cleaned = self.clean_text(text)

        tokens = self.tokenize(cleaned)

        return {

            "original_text": text,

            "cleaned_text": cleaned,

            "tokens": tokens,

            "token_count": len(tokens)

        }


# -------------------------------------------------------
# Testing
# -------------------------------------------------------

if __name__ == "__main__":

    preprocessor = TextPreprocessor()

    sample = "I am really frustrated!! I don't understand this machine learning code."

    result = preprocessor.preprocess(sample)

    print(result)