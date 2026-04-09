import pandas as pd
import joblib
from Labeling_data_ingestion.data_handler.import_data import load_data 
df = load_data(r"data/testing_data.csv")
df = df.dropna(subset=['theme'])

df1 = load_data(r"data/hand_named.csv")



pipeline = joblib.load("trained_models/pipeline.rfk")
replace = df["relevant"].isna()
X = df.loc[replace, "title"] + " " + df.loc[replace, "theme"]
df.loc[replace, "relevant"] = pipeline.predict(X)


df.to_csv("data/data_newly_Labeled.csv", index=False)




