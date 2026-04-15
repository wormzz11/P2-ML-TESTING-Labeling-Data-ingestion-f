from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from Labeling_data_ingestion.models.sklearn_models.threshold_classifier import ThresholdClassifier
from Labeling_data_ingestion.models.sklearn_models.transformer import MiniLmVectorizer
import joblib
from pathlib import Path

def train_test(df, test_size, seed = 40):

    X = df["title"] + " " + df["theme"] 
    y = df["relevant"]
    
    return train_test_split(X, y, random_state=seed, test_size=test_size)



def train_tfidf(model, X_train ,y_train,threshold = 0.45):
    
    if X_train is None or y_train is None:
        raise ValueError("Training data is None")

    wrapped = ThresholdClassifier(model, threshold=threshold)

    pipe = Pipeline([(
        "vectorizer", 
        TfidfVectorizer(
            stop_words="english", 
            ngram_range=(1,2),
            sublinear_tf=True
            )),
            ("model", wrapped)
    ])

    pipe.fit(X_train, y_train)

    return pipe




def train_transformer(model, X_train, y_train, threshold = 0.26 ):

    if X_train is None or y_train is None:
        raise ValueError("Training data is None")

    wrapped = ThresholdClassifier(model, threshold=threshold)

    pipe = Pipeline([(
        "vectorizer", 
        MiniLmVectorizer()),
            ("model", wrapped)
    ])

    pipe.fit(X_train, y_train)

    return pipe



def evaluate(pipe, X_test, y_test, threshold = 0.26):
    proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report
    }



def save_model(pipe, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)