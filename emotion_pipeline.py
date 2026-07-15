"""
============================================================
EMOTION PIPELINE
Epic 3 - Story 1
AI Learning Assistant
============================================================
"""

import os
import pickle
import warnings
import numpy as np
import tensorflow as tf
import torch

from src.emotion_predictor import EmotionPredictor
from src.bert_classifier import BERTEmotionClassifier
from src.keyword_enhancer import KeywordEnhancer
from src.mixed_emotion_detector import MixedEmotionDetector

warnings.filterwarnings("ignore")


class EmotionPipeline:

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING EMOTION PIPELINE")
        print("=" * 60)

        # --------------------------------------------------
        # Project Paths
        # --------------------------------------------------

        self.PROJECT_ROOT = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.ARTIFACT_DIR = os.path.join(
            self.PROJECT_ROOT,
            "artifacts"
        )

        self.MODEL_DIR = os.path.join(
            self.PROJECT_ROOT,
            "models"
        )

        self.BILSTM_MODEL_PATH = os.path.join(
            self.MODEL_DIR,
            "bilstm",
            "bilstm_student_adaptive.keras"
        )

        self.BERT_MODEL_PATH = os.path.join(
            self.MODEL_DIR,
            "bert_emotion_model_final"
        )

        # --------------------------------------------------
        # Configuration
        # --------------------------------------------------

        self.MAX_SEQUENCE_LENGTH = 80

        self.device = torch.device(

            "cuda"

            if torch.cuda.is_available()

            else

            "cpu"

        )

        print(f"Using Device : {self.device}")

        # --------------------------------------------------
        # Helper Classes
        # --------------------------------------------------

        self.preprocessor = TextPreprocessor()

        self.keyword_enhancer = KeywordEnhancer()

        # --------------------------------------------------
        # Models
        # --------------------------------------------------

        self.tokenizer = None

        self.label_encoder = None

        self.bilstm_model = None

        self.bert_tokenizer = None

        self.bert_model = None

        # --------------------------------------------------
        # Load Everything
        # --------------------------------------------------

        self.load_models()
            # ==========================================================
    # LOAD ALL MODELS AND ARTIFACTS
    # ==========================================================

    def load_models(self):

        print("\n" + "=" * 60)
        print("LOADING MODELS")
        print("=" * 60)

        # ------------------------------------------------------
        # Load Keras Tokenizer
        # ------------------------------------------------------

        tokenizer_path = os.path.join(
            self.ARTIFACT_DIR,
            "tokenizer.pkl"
        )

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        print("✔ Keras Tokenizer Loaded")

        # ------------------------------------------------------
        # Load Label Encoder
        # ------------------------------------------------------

        label_encoder_path = os.path.join(
            self.ARTIFACT_DIR,
            "label_encoder.pkl"
        )

        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        print("✔ Label Encoder Loaded")

        print("\nEmotion Classes:")

        for i, emotion in enumerate(self.label_encoder.classes_):
            print(f"{i} -> {emotion}")

        # ------------------------------------------------------
        # Load Student Adaptive BiLSTM
        # ------------------------------------------------------

        print("\nLoading Student Adaptive BiLSTM...")

        self.bilstm_model = tf.keras.models.load_model(
            self.BILSTM_MODEL_PATH
        )

        print("✔ Student Adaptive BiLSTM Loaded")

        # ------------------------------------------------------
        # Load BERT Tokenizer
        # ------------------------------------------------------

        print("\nLoading BERT Tokenizer...")

        self.bert_tokenizer = BertTokenizerFast.from_pretrained(
            self.BERT_MODEL_PATH
        )

        print("✔ BERT Tokenizer Loaded")

        # ------------------------------------------------------
        # Load BERT Model
        # ------------------------------------------------------

        print("\nLoading Fine-tuned BERT...")

        self.bert_model = BertForSequenceClassification.from_pretrained(
            self.BERT_MODEL_PATH
        )

        self.bert_model.to(self.device)

        self.bert_model.eval()

        print("✔ Fine-tuned BERT Loaded")

        print("\n" + "=" * 60)
        print("ALL MODELS LOADED SUCCESSFULLY")
        print("=" * 60)
        
            # ==========================================================
    # PREPROCESS FOR BILSTM
    # ==========================================================

    def prepare_bilstm_input(self, text):

        cleaned_text = self.preprocessor.clean_text(text)

        sequence = self.tokenizer.texts_to_sequences(
            [cleaned_text]
        )

        padded_sequence = pad_sequences(
            sequence,
            maxlen=self.MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post"
        )

        return padded_sequence

    # ==========================================================
    # BILSTM PREDICTION
    # ==========================================================

    def predict_bilstm(self, text):

        sequence = self.prepare_bilstm_input(text)

        probabilities = self.bilstm_model.predict(
            sequence,
            verbose=0
        )[0]

        predicted_index = np.argmax(probabilities)

        predicted_emotion = self.label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = float(probabilities[predicted_index])

        return {

            "emotion": predicted_emotion,

            "confidence": confidence,

            "probabilities": probabilities

        }

    # ==========================================================
    # PREPROCESS FOR BERT
    # ==========================================================

    def prepare_bert_input(self, text):

        cleaned_text = self.preprocessor.clean_text(text)

        encoded = self.bert_tokenizer(

            cleaned_text,

            truncation=True,

            padding="max_length",

            max_length=self.MAX_SEQUENCE_LENGTH,

            return_tensors="pt"

        )

        return encoded

    # ==========================================================
    # BERT PREDICTION
    # ==========================================================

    def predict_bert(self, text):

        encoded = self.prepare_bert_input(text)

        input_ids = encoded["input_ids"].to(self.device)

        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():

            outputs = self.bert_model(

                input_ids=input_ids,

                attention_mask=attention_mask

            )

        logits = outputs.logits

        probabilities = torch.softmax(

            logits,

            dim=1

        ).cpu().numpy()[0]

        predicted_index = np.argmax(probabilities)

        predicted_emotion = self.label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = float(probabilities[predicted_index])

        return {

            "emotion": predicted_emotion,

            "confidence": confidence,

            "probabilities": probabilities

        }
        # ==========================================================
    # COMBINE BILSTM + BERT PREDICTIONS
    # ==========================================================

    def combine_predictions(self, bilstm_probs, bert_probs):

        """
        Weighted Ensemble

        BERT receives slightly higher weight because
        it generally captures context better.
        """

        bilstm_weight = 0.40
        bert_weight = 0.60

        combined = (

            bilstm_weight * bilstm_probs +

            bert_weight * bert_probs

        )

        combined = combined / np.sum(combined)

        return combined

    # ==========================================================
    # APPLY KEYWORD ENHANCEMENT
    # ==========================================================

    def apply_keyword_enhancement(self, probabilities, text):

        enhanced_probs, keyword_scores = self.keyword_enhancer.enhance(

            probabilities,

            text,

            self.label_encoder.classes_

        )

        return enhanced_probs, keyword_scores

    # ==========================================================
    # DETECT MIXED EMOTIONS
    # ==========================================================

    def detect_mixed_emotions(self, probabilities):

        classes = self.label_encoder.classes_

        sorted_indices = np.argsort(probabilities)[::-1]

        primary_index = sorted_indices[0]
        secondary_index = sorted_indices[1]

        primary_emotion = classes[primary_index]
        secondary_emotion = classes[secondary_index]

        primary_confidence = float(probabilities[primary_index])
        secondary_confidence = float(probabilities[secondary_index])

        mixed = False

        if abs(primary_confidence - secondary_confidence) < 0.15:

            mixed = True

        return {

            "primary_emotion": primary_emotion,

            "secondary_emotion": secondary_emotion,

            "primary_confidence": primary_confidence,

            "secondary_confidence": secondary_confidence,

            "mixed_emotions": mixed

        }

    # ==========================================================
    # CREATE PROBABILITY DICTIONARY
    # ==========================================================

    def create_probability_dictionary(self, probabilities):

        result = {}

        for emotion, probability in zip(

            self.label_encoder.classes_,

            probabilities

        ):

            result[emotion] = round(

                float(probability),

                4

            )

        return result
        # ==========================================================
    # MAIN PREDICTION PIPELINE
    # ==========================================================

    def predict(self, text):

        if text is None or str(text).strip() == "":

            raise ValueError("Input text cannot be empty.")

        # ---------------------------------------------
        # Text preprocessing
        # ---------------------------------------------

        cleaned_text = self.preprocessor.clean_text(text)

        # ---------------------------------------------
        # Individual model predictions
        # ---------------------------------------------

        bilstm_result = self.predict_bilstm(cleaned_text)

        bert_result = self.predict_bert(cleaned_text)

        # ---------------------------------------------
        # Ensemble prediction
        # ---------------------------------------------

        combined_probabilities = self.combine_predictions(

            bilstm_result["probabilities"],

            bert_result["probabilities"]

        )

        # ---------------------------------------------
        # Keyword enhancement
        # ---------------------------------------------

        enhanced_probabilities, keyword_scores = self.apply_keyword_enhancement(

            combined_probabilities,

            cleaned_text

        )

        # ---------------------------------------------
        # Mixed emotion detection
        # ---------------------------------------------

        mixed_result = self.detect_mixed_emotions(
            enhanced_probabilities
        )

        probability_dictionary = self.create_probability_dictionary(
            enhanced_probabilities
        )

        return {

            "input_text": text,

            "cleaned_text": cleaned_text,

            "primary_emotion":
                mixed_result["primary_emotion"],

            "secondary_emotion":
                mixed_result["secondary_emotion"],

            "confidence":
                round(
                    mixed_result["primary_confidence"],
                    4
                ),

            "secondary_confidence":
                round(
                    mixed_result["secondary_confidence"],
                    4
                ),

            "mixed_emotions":
                mixed_result["mixed_emotions"],

            "probabilities":
                probability_dictionary,

            "keyword_scores":
                keyword_scores,

            "bilstm_prediction":
                bilstm_result["emotion"],

            "bert_prediction":
                bert_result["emotion"]

        }
        # ==========================================================
# TESTING
# ==========================================================

if __name__ == "__main__":

    pipeline = EmotionPipeline()

    samples = [

        "I am completely confused about neural networks.",

        "This assignment is so frustrating. I have tried everything.",

        "Wow! I finally understood backpropagation.",

        "Machine learning is becoming boring.",

        "I wonder how transformers actually work."

    ]

    print("\n" + "=" * 60)
    print("TESTING EMOTION PIPELINE")
    print("=" * 60)

    for sentence in samples:

        print("\n" + "-" * 60)

        result = pipeline.predict(sentence)

        print("Input :", sentence)

        print("Primary Emotion :", result["primary_emotion"])

        print("Secondary Emotion :", result["secondary_emotion"])

        print("Confidence :", result["confidence"])

        print("Mixed Emotion :", result["mixed_emotions"])

        print("BiLSTM :", result["bilstm_prediction"])

        print("BERT :", result["bert_prediction"])

        print("Probabilities")

        for emotion, probability in result["probabilities"].items():

            print(f"{emotion:12} : {probability:.4f}")