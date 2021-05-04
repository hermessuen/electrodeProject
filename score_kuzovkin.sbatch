#!/bin/bash
#SBATCH --job-name=neural-array
#SBATCH --array=0-8
#SBATCH --time=92:00:00
#SBATCH --ntasks=1
#SBATCH --mem=400000

i=0
for model in bagnet17 \
bagnet33 \
bagnet9 \
fixres_resnext101_32x48d_wsl \
resnet18-contrastive_multiview \
resnet50-SIN \
resnet50-SIN_IN \
resnet50-SIN_IN_IN; do
	model_list[$i]="$model"
	i=$[$i+1]
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"

python score_models.py "${model_list[$SLURM_ARRAY_TASK_ID]}"