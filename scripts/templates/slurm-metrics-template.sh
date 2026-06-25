#!/bin/bash

#SBATCH --time=03:00:00         # hours minutes seconds
#SBATCH --gpus=1
#SBATCH --output $PROJECTDIR/DatasetSimilarity/slurm_cave/logs/metrics/{{experiment_name}}-%A_%a.out
#SBATCH --array=0-{{num_tasks}}


echo "Activating the Python Environment"
source $PROJECTDIR/DatasetSimilarity/ARC-Dataset-Similarity/.venv/bin/activate

echo "Running the metrics script"
python $PROJECTDIR/DatasetSimilarity/ARC-Dataset-Similarity/scripts/run_metrics.py \
    --config {{experiment_name}}/${SLURM_ARRAY_TASK_ID}
echo "Finished running the metrics script"
date
