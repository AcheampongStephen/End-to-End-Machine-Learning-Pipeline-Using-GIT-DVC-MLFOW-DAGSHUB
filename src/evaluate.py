import pandas as pd
import os
import pickle
import yaml
import mlflow
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/AcheampongStephen/End-to-End-Machine-Learning-Pipeline-Using-GIT-DVC-MLFOW-DAGSHUB.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "AcheampongStephen"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9b7471a68fbc760387135ad6eb899bde0456a4f0"

# Load the parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    # Load the data
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/AcheampongStephen/End-to-End-Machine-Learning-Pipeline-Using-GIT-DVC-MLFOW-DAGSHUB.mlflow")

    # Load the model
    model = pickle.load(open(model_path, "rb"))


    # Predict and evaluate the model
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Log the evaluation metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)

    print(f"Model Accuracy: {accuracy}")



if __name__ == "__main__":
    evaluate(params["data"],params["model"])