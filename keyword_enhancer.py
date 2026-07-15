"""
============================================================
KEYWORD ENHANCER
Epic 3 - Story 1
============================================================
"""

import numpy as np


class KeywordEnhancer:

    def __init__(self):

        self.emotion_keywords = {

            "Frustrated": [
                "frustrated", "frustrating", "annoying", "angry",
                "hate", "difficult", "stuck", "wrong answer",
                "keep getting", "unnecessarily complicated",
                "tried", "failed", "error", "bug"
            ],

            "Curious": [
                "why", "how", "what", "curious",
                "wonder", "interested", "learn",
                "know more", "want to know",
                "explore", "could we",
                "what happens", "question"
            ],

            "Confident": [
                "easy", "amazing", "great",
                "excellent", "good", "awesome",
                "perfect", "solved", "got it",
                "clear now", "finally",
                "understand clearly"
            ],

            "Bored": [
                "boring", "bored", "tired",
                "repetitive", "dull",
                "not engaging",
                "didn't feel engaging",
                "too basic",
                "losing interest"
            ],

            "Confused": [
                "confused", "lost",
                "unclear",
                "don't understand",
                "doesn't make sense",
                "missing",
                "incomplete",
                "unsure"
            ]

        }

    # ----------------------------------------------------

    def enhance(self, probabilities, text, classes):

        """
        probabilities : numpy array from model
        text          : cleaned text
        classes       : label encoder classes
        """

        probs = probabilities.copy()

        text = text.lower()

        emotion_scores = {}

        for emotion, keywords in self.emotion_keywords.items():

            score = 0

            for keyword in keywords:

                if keyword in text:

                    # Strong boost for explicit emotion words
                    if keyword in [
                        "frustrated",
                        "curious",
                        "confident",
                        "bored",
                        "boring",
                        "confused"
                    ]:

                        score += 10

                    else:

                        score += 2

            emotion_scores[emotion] = score

        max_score = max(emotion_scores.values())

        if max_score > 0:

            for emotion, score in emotion_scores.items():

                if score == max_score:

                    idx = list(classes).index(emotion)

                    probs[idx] *= (1 + score * 3)

            probs = probs / np.sum(probs)

        return probs, emotion_scores


# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    enhancer = KeywordEnhancer()

    probabilities = np.array([
        0.20,
        0.20,
        0.20,
        0.20,
        0.20
    ])

    classes = [
        "Bored",
        "Confident",
        "Confused",
        "Curious",
        "Frustrated"
    ]

    text = "I am really frustrated because I don't understand this code."

    new_probs, scores = enhancer.enhance(
        probabilities,
        text,
        classes
    )

    print("Keyword Scores")

    print(scores)

    print()

    print("Updated Probabilities")

    print(new_probs)