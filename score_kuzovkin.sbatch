#!/bin/bash
#SBATCH --job-name=neural-array
#SBATCH --array=0-2
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=150G

i=0
for model in alexnet densenet-169 resnet-18; do
	model_list[$i]="$model"
	i=$[$i+1]
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"

python score_models.py "${model_list[$SLURM_ARRAY_TASK_ID]}"