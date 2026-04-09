from Labeling_data_ingestion.models.sklearn_models.sk_models import logistic_model
from Labeling_data_ingestion.train.train_sklearn import train

model = train(logistic_model(), 0.3)
    
