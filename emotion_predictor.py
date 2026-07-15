"""
============================================================
EMOTION PREDICTOR
Epic 3 - Story 2
============================================================
"""

import os
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.text_preprocessor import TextPreprocessor

class EmotionPredictor:

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING BILSTM PREDICTOR")
        print("=" * 60)

        self.PROJECT_ROOT = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.MAX_SEQUENCE_LENGTH = 80

        self.preprocessor = TextPreprocessor()

        # ----------------------------------------------------
        # Load Tokenizer
        # ----------------------------------------------------

        tokenizer_path = os.path.join(
            self.PROJECT_ROOT,
            "artifacts",
            "tokenizer.pkl"
        )

        with open(tokenizer_path, "rb") as f:

            self.tokenizer = pickle.load(f)

        print("✔ Tokenizer Loaded")

        # ----------------------------------------------------
        # Load Label Encoder
        # ----------------------------------------------------

        encoder_path = os.path.join(
            self.PROJECT_ROOT,
            "artifacts",
            "label_encoder.pkl"
        )

        with open(encoder_path, "rb") as f:

            self.label_encoder = pickle.load(f)

        self.classes = list(self.label_encoder.classes_)

        print("✔ Label Encoder Loaded")

        # ----------------------------------------------------
        # Load BiLSTM
        # ----------------------------------------------------

        model_path = os.path.join(

            self.PROJECT_ROOT,

            "models",

            "bilstm",

            "bilstm_student_adaptive.keras"

        )

        self.model = tf.keras.models.load_model(model_path)

        print("✔ Student Adaptive BiLSTM Loaded")
            # ==========================================================
    # PREPARE INPUT SEQUENCE
    # ==========================================================

    def prepare_sequence(self, text):

        cleaned_text = self.preprocessor.clean_text(text)

        sequence = self.tokenizer.texts_to_sequences([cleaned_text])

        padded_sequence = pad_sequences(
            sequence,
            maxlen=self.MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post"
        )

        return cleaned_text, padded_sequence

    # ==========================================================
    # PREDICT EMOTION
    # ==========================================================

    def predict(self, text):

        cleaned_text, sequence = self.prepare_sequence(text)

        probabilities = self.model.predict(
            sequence,
            verbose=0
        )[0]

        predicted_index = np.argmax(probabilities)

        predicted_emotion = self.classes[predicted_index]

        confidence = float(probabilities[predicted_index])

        probability_dict = {}

        for emotion, prob in zip(self.classes, probabilities):

            probability_dict[emotion] = round(float(prob), 4)

        return {

            "emotion": predicted_emotion,

            "confidence": round(confidence,4),

            "scores": probability_dict,

            "cleaned_text": cleaned_text

        }

       # ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    predictor = EmotionPredictor()

    sample = "I am confident that I solved this coding problem."

    result = predictor.predict(sample)

    print("\n" + "=" * 60)
    print("BILSTM PREDICTION")
    print("=" * 60)

    print("Cleaned Text :", result["cleaned_text"])
    print("Emotion :", result["emotion"])
    print("Confidence :", result["confidence"])

    print("\nScores")

    for emotion, score in result["scores"].items():
        print(f"{emotion:12}: {score:.4f}")

    print("\nSoftmax Sum :", sum(result["scores"].values()))