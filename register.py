import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, f1_score

#Load Data

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("wine-classifer")

mlflow.set_experiment("wine-classifier")

for n_est in [50, 100, 200]:
    for depth in [3, 5, 7]:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=n_est, max_depth=depth)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_param("n_estimators", n_est)
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

print("All runs complete")
  



print("Model registered and moved to Staging")
