"""
============================================================
CSV MANAGER
Epic 3 - Story 6
============================================================
"""

import os
import pandas as pd
from datetime import datetime


class CSVManager:

    def __init__(self):

        self.examples_file = "emotion_response_examples.csv"

        self.mapping_file = "emotion_response_mapping.csv"

    def save_interaction(

        self,

        field,

        text,

        emotion,

        confidence,

        response

    ):

        new_row = {

            "text": text,

            "emotion": emotion,

            "confidence": confidence,

            "response": response,

            "field": field,

            "timestamp": datetime.now().isoformat()

        }

        # -----------------------------

        if os.path.exists(self.examples_file):

            try:
                df = pd.read_csv(self.examples_file)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                df = pd.DataFrame([new_row])

            df = pd.concat(

                [df, pd.DataFrame([new_row])],

                ignore_index=True

            )

        else:

            df = pd.DataFrame([new_row])

        df.to_csv(

            self.examples_file,

            index=False

        )

        # -----------------------------
        # Update Mapping File
        # -----------------------------

        if os.path.exists(self.mapping_file):

            mapping_df = pd.read_csv(self.mapping_file)

        else:

            mapping_df = pd.DataFrame(

                columns=[

                    "emotion",

                    "response"

                ]

            )

        if not (

            (mapping_df["emotion"] == emotion) &

            (mapping_df["response"] == response)

        ).any():

            mapping_df = pd.concat(

                [

                    mapping_df,

                    pd.DataFrame([{

                        "emotion": emotion,

                        "response": response

                    }])

                ],

                ignore_index=True

            )

            mapping_df.to_csv(

                self.mapping_file,

                index=False

            )

        return True

    def get_example_response(self, text, emotion=None):
        if not os.path.exists(self.examples_file):
            return None

        try:
            df = pd.read_csv(self.examples_file)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return None

        if df.empty:
            return None

        if emotion and "emotion" in df.columns:
            emotion_matches = df[df["emotion"] == emotion]
            if not emotion_matches.empty:
                return emotion_matches.iloc[-1]["response"]

        if "text" in df.columns:
            text_lower = text.lower().strip()
            for _, row in df.iterrows():
                if str(row["text"]).lower().strip() == text_lower:
                    return row["response"]

        return None

    def count_examples(self):
        if not os.path.exists(self.examples_file):
            return 0

        try:
            return len(pd.read_csv(self.examples_file))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return 0

if __name__ == "__main__":

    manager = CSVManager()

    manager.save_interaction(

        field="Machine Learning",

        text="I am frustrated with neural networks.",

        emotion="Frustrated",

        confidence=0.96,

        response="Let's solve it together."

    )

    print("Interaction Saved Successfully.")
