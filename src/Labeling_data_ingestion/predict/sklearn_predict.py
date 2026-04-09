import joblib
from src.Labeling_data_ingestion.train.train_sklearn import train


train()


pipe = joblib.load("trained_models/pipeline.rfk")

