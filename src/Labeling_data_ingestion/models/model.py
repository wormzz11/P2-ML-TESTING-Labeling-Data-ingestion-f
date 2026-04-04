import torch
import pandas as pd
from Labeling_data_ingestion.config import DATA_PATH
from  Labeling_data_ingestion.data_handler.import_data import load_data

df = load_data(DATA_PATH)
print(df.head())

