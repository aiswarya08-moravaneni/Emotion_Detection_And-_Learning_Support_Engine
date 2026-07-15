"""
============================================================
BERT EMOTION CLASSIFIER
Epic 3 - Story 3
============================================================
"""

import os
import pickle
import numpy as np
import torch

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification
)
from src.text_preprocessor import TextPreprocessor


class BERTEmotionClassifier:

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING BERT CLASSIFIER")
        print("=" * 60)

        self.PROJECT_ROOT = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.preprocessor = TextPreprocessor()

        self.load_model()

    def load_model(self):

        model_path = os.path.join(
            self.PROJECT_ROOT,
            "models",
            "bert_emotion_model_final"
        )

        print("\nLoading Tokenizer...")

        self.tokenizer = BertTokenizerFast.from_pretrained(
            model_path
        )

        print("✔ Tokenizer Loaded")

        print("\nLoading Fine-tuned BERT...")

        self.model = BertForSequenceClassification.from_pretrained(
            model_path
        )

        self.model.to(self.device)

        self.model.eval()

        print("✔ BERT Loaded")

        encoder_path = os.path.join(
            self.PROJECT_ROOT,
            "artifacts",
            "label_encoder.pkl"
        )

        with open(encoder_path, "rb") as f:

            self.label_encoder = pickle.load(f)

        self.classes = list(self.label_encoder.classes_)

        self.class_weights = {

            "Bored": 1.2,
            "Confident": 1.8,
            "Confused": 0.6,
            "Curious": 1.0,
            "Frustrated": 1.4

        }

        print("✔ Label Encoder Loaded")

    def predict(self, text):

        cleaned_text = self.preprocessor.clean_text(text)

        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=80
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            ).cpu().numpy()[0]

        # -------------------------------------------------
        # Apply Class Weights
        # -------------------------------------------------

        weights = np.array([

            self.class_weights["Bored"],

            self.class_weights["Confident"],

            self.class_weights["Confused"],

            self.class_weights["Curious"],

            self.class_weights["Frustrated"]

        ])

        weighted_probabilities = probabilities * weights

        # -------------------------------------------------
        # Keyword Adjustment
        # -------------------------------------------------

        text_lower = cleaned_text.lower()

        confidence_keywords = [

            "confident",
            "easy",
            "clear",
            "understand",
            "solved",
            "finally",
            "got it",
            "correct"

        ]

        confusion_keywords = [

            "confused",
            "don't understand",
            "unclear",
            "lost",
            "stuck",
            "difficult",
            "problem"

        ]

        if any(keyword in text_lower for keyword in confidence_keywords):

            weighted_probabilities[1] *= 2.5
            weighted_probabilities[2] *= 0.3

        elif any(keyword in text_lower for keyword in confusion_keywords):

            weighted_probabilities[2] *= 2.0

        # -------------------------------------------------
        # Normalize Again
        # -------------------------------------------------

        weighted_probabilities = (
            weighted_probabilities /
            np.sum(weighted_probabilities)
        )

        prediction = np.argmax(weighted_probabilities)

        emotion = self.classes[prediction]

        confidence = float(weighted_probabilities[prediction])

        probability_dict = {}

        for emotion_name, score in zip(
            self.classes,
            weighted_probabilities
        ):

            probability_dict[emotion_name] = round(
                float(score),
                4
            )

        return {

            "emotion": emotion,

            "confidence": round(confidence,4),

            "scores": probability_dict,

            "cleaned_text": cleaned_text

        }
# =====================================================
# TESTING
# =====================================================

if __name__ == "__main__":

    classifier = BERTEmotionClassifier()

    samples = [

        "I don't understand this algorithm.",

        "I solved the assignment successfully.",

        "I am curious about transformers.",

        "This lecture is boring.",

        "I am frustrated with this coding error."

    ]

    print("\n" + "=" * 60)
    print("BERT PREDICTIONS")
    print("=" * 60)

    for sample in samples:

        result = classifier.predict(sample)

        print("\n" + "-" * 60)

        print("Cleaned Text :", result["cleaned_text"])
        print("Emotion :", result["emotion"])
        print("Confidence :", result["confidence"])

        print("Scores")

        for emotion, score in result["scores"].items():
          print(f"{emotion:12}: {score:.4f}")