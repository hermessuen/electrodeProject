#!/bin/bash
#SBATCH --job-name=neural-array
#SBATCH --array=0-49
#SBATCH --time=92:00:00
#SBATCH --ntasks=1
#SBATCH --mem=300000

i=0
for model in mobilenet_v1_0.5_224 mobilenet_v1_0.5_192 mobilenet_v1_0.5_160 mobilenet_v1_0.5_128 \
mobilenet_v1_0.25_224 mobilenet_v1_0.25_192 mobilenet_v1_0.25_160 mobilenet_v1_0.25_128 mobilenet_v2_1.4_224 mobilenet_v2_1.3_224 \
mobilenet_v2_1.0_224 mobilenet_v2_1.0_192 mobilenet_v2_1.0_160 mobilenet_v2_1.0_128 mobilenet_v2_1.0_96 mobilenet_v2_0.75_224 \
mobilenet_v2_0.75_192 mobilenet_v2_0.75_160 mobilenet_v2_0.75_128 mobilenet_v2_0.75_96 mobilenet_v2_0.5_224 mobilenet_v2_0.5_192 \
mobilenet_v2_0.5_160 mobilenet_v2_0.5_128 mobilenet_v2_0.5_96 \
mobilenet_v2_0.35_224 \
mobilenet_v2_0.35_192 \
mobilenet_v2_0.35_160 \
mobilenet_v2_0.35_128 \
mobilenet_v2_0.35_96 \
resnet50-SIN \
resnet50-SIN_IN \
resnet50-SIN_IN_IN \
resnext101_32x48d_wsl \
resnext101_32x8d_wsl \
resnext101_32x16d_wsl \
resnext101_32x32d_wsl \
fixres_resnext101_32x48d_wsl \
resnet18-supervised \
resnet18-local_aggregation \
resnet18-instance_recognition \
resnet18-autoencoder \
resnet18-contrastive_predictive \
resnet18-colorization \
resnet18-relative_position \
resnet18-depth_prediction \
prednet \
resnet18-simclr \
resnet18-deepcluster resnet18-contrastive_multiview; do
	model_list[$i]="$model"
	i=$[$i+1]
done



echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"

python score_models.py "${model_list[$SLURM_ARRAY_TASK_ID]}"