#!/bin/bash

#SBATCH --time=03:00:00         # hours minutes seconds
#SBATCH --gpus=1
#SBATCH --output $PROJECTDIR/DatasetSimilarity/slurm_cave/logs/finetune/experiment_1_main-%A_%a.out
#SBATCH --array=0-8


echo "Activating the Python Environment"
source $PROJECTDIR/DatasetSimilarity/ARC-Dataset-Similarity/.venv/bin/activate

echo "Running the fine-tuning script"
python $PROJECTDIR/DatasetSimilarity/ARC-Dataset-Similarity/scripts/finetune.py \
    --config experiment_1_main/${SLURM_ARRAY_TASK_ID}
echo "Finished running the fine-tuning script"
date
