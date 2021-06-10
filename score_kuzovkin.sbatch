#!/bin/bash
#SBATCH --job-name=neural-array
#SBATCH --array=0-35
#SBATCH --time=92:00:00
#SBATCH --ntasks=1
#SBATCH --mem=400000

i=0
for model in mobilenet_v2_1.0_128 \
nasnet_large \
mobilenet_v1_0.25_128 \
nasnet_mobile \
mobilenet_v2_1.0_224 \
resnet18-relative_position \
resnet18-deepcluster \
mobilenet_v2_0.75_160 \
inception_v1 \
mobilenet_v1_0.25_192 \
resnet18-simclr \
resnet-152_v2 \
mobilenet_v2_1.3_224 \
pnasnet_large \
mobilenet_v2_0.5_224 \
mobilenet_v2_1.0_160 \
resnet-101_v1 \
mobilenet_v2_0.75_224 \
resnet18-depth_prediction \
mobilenet_v2_0.5_192 \
mobilenet_v2_1.0_96 \
mobilenet_v2_0.75_96 \
resnet18-local_aggregation \
prednet \
resnet-101_v2 \
mobilenet_v1_0.75_224 \
resnet18-supervised \
resnet18-autoencoder \
mobilenet_v1_1.0_192 \
mobilenet_v1_1.0_128 \
mobilenet_v2_0.75_192 \
resnet18-contrastive_predictive \
mobilenet_v2_0.75_128 \
resnet18-instance_recognition \
fixres_resnext101_32x48d_wsl \
resnet18-colorization; do
	model_list[$i]="$model"
	i=$[$i+1]
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"

python score_models.py "${model_list[$SLURM_ARRAY_TASK_ID]}"