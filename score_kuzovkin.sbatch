#!/bin/bash
#SBATCH --job-name=neural-array
#SBATCH --array=0-17
#SBATCH --time=92:00:00
#SBATCH --ntasks=1
#SBATCH --mem=400000

i=0
for model in CORnet-S \
CORnet-R \
CORnet-Z \
CORnet-R2; do
	model_list[$i]="$model"
	i=$[$i+1]
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"

python score_models.py "${model_list[$SLURM_ARRAY_TASK_ID]}"