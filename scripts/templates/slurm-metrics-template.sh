#!/bin/bash

#SBATCH --time={{time}}        # hours minutes seconds
#SBATCH --gpus=1
#SBATCH --output {{root}}DatasetSimilarity/slurm_cave/logs/metrics/{{experiment_name}}-%A_%a.out
#SBATCH --error {{root}}DatasetSimilarity/slurm_cave/logs/metrics/{{experiment_name}}-%A_%a.err
#SBATCH --array=0-{{num_tasks}}

echo "--- MLFLOW ENVIRONMENT ---"
echo MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME
echo MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
echo "Server Status: $(curl -s $MLFLOW_TRACKING_URI/health)"
echo "Authenticated User: $(curl -s $MLFLOW_TRACKING_URI/api/2.0/mlflow/users/get?username=$MLFLOW_TRACKING_USERNAME -u "$MLFLOW_TRACKING_USERNAME:$MLFLOW_TRACKING_PASSWORD")"

echo "--- KEOPS ---"  # Only needed if using GeomLoss on large samples
module load cudatoolkit
module load PrgEnv-gnu
export CUDA_PATH=$CUDA_HOME
export CXX=g++
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export KEOPS_CACHE_FOLDER=$SCRATCH/keops_cache/$(uname -m)

echo "Activating the Python Environment"
source $PROJECTDIR/DatasetSimilarity/ARC-Dataset-Similarity/.venv/bin/activate

echo "Running the metrics script"
python $PROJECTDIR/DatasetSimilarity/ARC-Dataset-Similarity/scripts/run_metrics.py \
    --config {{experiment_name}}/metrics_${SLURM_ARRAY_TASK_ID}
echo "Finished running the metrics script"
date
