#!/bin/bash
#SBATCH --job-name=neural-array
#SBATCH --array=0-8
#SBATCH --time=92:00:00
#SBATCH --ntasks=1
#SBATCH --mem=400000

i=0
for model in mobilenet_v2_0.35_128 \
mobilenet_v2_0.35_160 \
mobilenet_v2_0.35_192 \
mobilenet_v2_0.35_224 \
mobilenet_v2_0.35_96 \
mobilenet_v2_0.5_128 \
mobilenet_v2_0.5_160 \
mobilenet_v2_0.5_96; do
	model_list[$i]="$model"
	i=$[$i+1]
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"

python score_models.py "${model_list[$SLURM_ARRAY_TASK_ID]}"