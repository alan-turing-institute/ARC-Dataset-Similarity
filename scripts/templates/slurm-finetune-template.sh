#!/bin/bash

#SBATCH --time=03:00:00         # hours minutes seconds
#SBATCH --gpus=1
#SBATCH --output $PROJECTDIR/DatasetSimilarity/slurm_cave/logs/finetune/{{experiment_name}}-%A_%a.out
#SBATCH --array=0-{{num_tasks}}


echo "Activating the Python Environment"
source $PROJECTDIR/DatasetSimilarity/ARC-Dataset-Similarity/.venv/bin/activate

echo "Running the fine-tuning script"
python $PROJECTDIR/DatasetSimilarity/ARC-Dataset-Similarity/scripts/finetune.py \
    --config {{experiment_name}}/${SLURM_ARRAY_TASK_ID}
echo "Finished running the fine-tuning script"
date
