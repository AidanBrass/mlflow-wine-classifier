import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
run_id = "509144916ad34575bea066be052b8abc"
artifacts = client.list_artifacts(run_id)
for a in artifacts:
    print(a.path)