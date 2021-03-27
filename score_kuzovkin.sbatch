#!/bin/bash
#SBATCH --job-name=neural-array
#SBATCH --array=0-1
#SBATCH --time=288:00:00
#SBATCH --ntasks=1
#SBATCH --mem=150000

i=0
for model in alexnet resnet-18; do
	model_list[$i]="$model"
	i=$[$i+1]
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"

python score_models.py "${model_list[$SLURM_ARRAY_TASK_ID]}"