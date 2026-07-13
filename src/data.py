"""Load the IMDB Large Movie Review Dataset (Maas et al., 2011).

50,000 movie reviews labeled positive (1) or negative (0),
split 25k train / 25k test, perfectly balanced.
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_imdb(cache_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with columns: text, label.

    label: 0 = negative, 1 = positive.
    """
    dataset = load_dataset("stanfordnlp/imdb", cache_dir=str(cache_dir))
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_imdb()
    print(f"train: {len(train_df)} rows, test: {len(test_df)} rows")
    print(train_df["label"].value_counts())
