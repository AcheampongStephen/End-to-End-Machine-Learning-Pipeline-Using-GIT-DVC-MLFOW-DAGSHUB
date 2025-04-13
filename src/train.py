import pandas as pd
import os
import yaml
import pickle
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/AcheampongStephen/End-to-End-Machine-Learning-Pipeline-Using-GIT-DVC-MLFOW-DAGSHUB.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "AcheampongStephen"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9b7471a68fbc760387135ad6eb899bde0456a4f0"


def hyperparameter_tuning(X_train,y_train,param_grid):
    rf=RandomForestClassifier()
    grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,y_train)
    return grid_search


# Load the parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data= pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/AcheampongStephen/End-to-End-Machine-Learning-Pipeline-Using-GIT-DVC-MLFOW-DAGSHUB.mlflow")

    ## Start an MLflow run
    with mlflow.start_run():
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        signature= infer_signature(X_train, y_train)

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        # Get the best parameters
        best_model = grid_search.best_estimator_

        # Predict and evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy_Score: {accuracy}")

        # Log additional metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_param("max_depth", grid_search.best_params_["max_depth"])
        mlflow.log_param("min_samples_split", grid_search.best_params_["min_samples_split"])
        mlflow.log_param("min_samples_leaf", grid_search.best_params_["min_samples_leaf"])
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("model_name", "RandomForest")
        mlflow.log_param("model_version", "v1.0")

        # Log the confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")  


        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            # Model registry is used in a remote server
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Model")
        else:
            # Model registry is used locally
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        ## Save the model locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        filename=model_path
        pickle.dump(best_model,open(filename,"wb"))
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train(
        params["data"],
        params["model"],
        params["random_state"],
        params["n_estimators"],
        params["max_depth"]
    )