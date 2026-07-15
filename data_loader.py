import pandas as pd
from pathlib import Path

# Base data folder
DATA_DIR = Path("data/raw")


def load_goemotions():
    """
    Load and merge the three GoEmotions CSV files.
    """

    files = [
        DATA_DIR / "goemotions_1.csv",
        DATA_DIR / "goemotions_2.csv",
        DATA_DIR / "goemotions_3.csv",
    ]

    dfs = []

    for file in files:
        print(f"Loading {file.name}...")
        df = pd.read_csv(file)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    print("\nGoEmotions loaded successfully!")
    print(f"Total rows: {merged_df.shape[0]}")
    print(f"Total columns: {merged_df.shape[1]}")

    return merged_df