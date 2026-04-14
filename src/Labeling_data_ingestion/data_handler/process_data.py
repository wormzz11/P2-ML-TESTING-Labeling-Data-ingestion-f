import os 
import pandas as pd

def load_data(PATH):
    return pd.read_csv(PATH)


def append_csv(subset, path):
    subset.to_csv(path, mode='a', index=False, header=not os.path.exists(path))


def build_dataset(df):

    return df[["title", "theme", "relevant"]].dropna()

def build_prediction_dataset(df):
    return df["title"] + " " + df["theme"]