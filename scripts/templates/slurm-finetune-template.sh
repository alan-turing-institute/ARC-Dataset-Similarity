#!/bin/bash

#SBATCH --time={{time}}         # hours minutes seconds
#SBATCH --gpus=1
#SBATCH --output {{root}}DatasetSimilarity/slurm_cave/logs/finetune/{{experiment_name}}-%A_%a.out
#SBATCH --error {{root}}DatasetSimilarity/slurm_cave/logs/finetune/{{experiment_name}}-%A_%a.err
#SBATCH --array=0-{{num_tasks}}

echo "--- MLFLOW ENVIRONMENT ---"
echo MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME
echo MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
echo "Server Status: $(curl -s $MLFLOW_TRACKING_URI/health)"
echo "Authenticated User: $(curl -s $MLFLOW_TRACKING_URI/api/2.0/mlflow/users/get?username=$MLFLOW_TRACKING_USERNAME -u "$MLFLOW_TRACKING_USERNAME:$MLFLOW_TRACKING_PASSWORD")"

echo "Warming MLflow container..."
for i in $(seq 1 20); do
  if curl -s -o /dev/null --max-time 60 "$MLFLOW_TRACKING_URI/health"; then
    echo "MLflow is awake."
    break
  fi
  echo "  attempt $i: still cold, retrying..."
  sleep 5
done

echo "Activating the Python Environment"
source $PROJECTDIR/DatasetSimilarity/ARC-Dataset-Similarity/.venv/bin/activate

echo "Running the fine-tuning script"
python $PROJECTDIR/DatasetSimilarity/ARC-Dataset-Similarity/scripts/finetune.py \
    --config {{experiment_name}}/finetune_${SLURM_ARRAY_TASK_ID}
echo "Finished running the fine-tuning script"
date
