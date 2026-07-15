"""
============================================================
MIXED EMOTION DETECTOR
Epic 3 - Story 4
============================================================
"""

class MixedEmotionDetector:

    def __init__(self, threshold=0.15):

        self.threshold = threshold

    def detect(self, scores):

        """
        scores:

        {
            "Bored":0.05,
            "Confident":0.42,
            "Confused":0.31,
            ...
        }
        """

        sorted_scores = sorted(

            scores.items(),

            key=lambda x: x[1],

            reverse=True

        )

        mixed = []

        for emotion, score in sorted_scores:

            if score >= self.threshold:

                mixed.append((emotion, score))

        if len(mixed) == 0:

            mixed.append(sorted_scores[0])

        return mixed
if __name__ == "__main__":

    detector = MixedEmotionDetector()

    scores = {

        "Bored":0.08,

        "Confident":0.46,

        "Confused":0.31,

        "Curious":0.07,

        "Frustrated":0.08

    }

    result = detector.detect(scores)

    print(result)