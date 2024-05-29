import subprocess
import mlflow
from mlflow.client import MlflowClient

workspace_dir = "C:\\Users\\billk\\workspace"

experiment_name = 'Default'
mlflow.set_experiment(experiment_name)

PORT = 8001  # REST API serving port
CONTAINER_NAME = "mlflow_example_model_serving"

# Get latest registered model
client = MlflowClient()
registered_models = client.search_registered_models()

## Sort registered models by latest version creation timestamp ##
latest_model = None
latest_version = None
latest_timestamp = 0

for model in registered_models:
    for version in model.latest_versions:
        if (
            version.current_stage == "Production"
            and version.creation_timestamp > latest_timestamp
        ):
            latest_model = model
            latest_version = version
            latest_timestamp = version.creation_timestamp

experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id
filter_string = f"tags.mlflow.runName = '{latest_model.name}'"
latest_run = client.search_runs(
    experiment_ids=[experiment_id], filter_string=filter_string
)

best_model_uri = f"{latest_run[0].info.artifact_uri}/model"

## Get best val loss model uri ##
# experiment = mlflow.get_experiment_by_name(experiment_name)
# experiment_id = experiment.experiment_id
# best_run_df = mlflow.search_runs(order_by=['metrics.Validation_Loss ASC'], max_results=1)
# if len(best_run_df.index) == 0:
#     raise Exception(f"Found no runs for experiment '{experiment_name}'")

# best_run = mlflow.get_run(best_run_df.at[0, 'run_id'])
# best_model_uri = f"{best_run.info.artifact_uri}/model"

# Log latest registered model's info
print("Best run info:")
print(f"Run id: {latest_run[0].info.run_id}")
print(f"Run parameters: {latest_run[0].data.params}")
print("Run score: Validation Loss = {:.4f}".format(latest_run[0].data.metrics['Validation_Loss']))
print(f"Run model URI: {best_model_uri}")

# remove current container if exists
subprocess.run(f"docker rm --force {CONTAINER_NAME}", shell=True, check=False, stdout=subprocess.DEVNULL)

docker_run_cmd = f"""
docker run
--name={CONTAINER_NAME}
--gpus all
--volume={workspace_dir.replace('\\', '/')}:/workspace
--workdir=/workspace/Speech_Recognition
--publish {PORT}:{PORT}
--interactive
--rm
mlflow
mlflow models serve --model-uri {best_model_uri} --host 0.0.0.0 --port {PORT} --workers 2 --no-conda
""".replace('\n', ' ').strip()
print(f"Running command:\n{docker_run_cmd}")

subprocess.run(docker_run_cmd, shell=True, check=True)

