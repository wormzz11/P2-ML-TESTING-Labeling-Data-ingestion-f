import pandas as pd
import joblib
from Labeling_data_ingestion.data_handler.process_data import load_data, append_csv
import os

df = load_data(r"data/unlabeled.csv")
df = df.dropna(subset=['theme'])



pipeline = joblib.load("trained_models/pipeline.rfk")
X = df["title"] + " " + df["theme"]

scores = pd.Series(pipeline.predict_proba(X)[:, 1], index=df.index)

certain_relevant = scores >= 0.6
certain_irrelevant = scores < 0.3
manual_review = ~certain_irrelevant & ~certain_relevant

df.loc[certain_relevant, "relevant"] = 1.0
df.loc[certain_irrelevant, "relevant"] = 0.0
df.loc[manual_review, "relevant"] = None



append_csv(df.loc[certain_irrelevant | certain_relevant], "data/certain_auto/auto_labeled.csv")
append_csv(df.loc[manual_review], "data/manual_review/manual_review.csv")












