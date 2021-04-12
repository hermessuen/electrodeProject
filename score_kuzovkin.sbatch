#!/bin/bash
#SBATCH --job-name=neural-array
#SBATCH --array=0-38
#SBATCH --time=92:00:00
#SBATCH --ntasks=1
#SBATCH --mem=400000

i=0
for model in CORnet-R CORnet-R2 CORnet-S CORnet-Z bagnet17 bagnet33 bagnet9 \
fixres_resnext101_32x48d_wsl \
inception_v2 \
inception_v3 \
mobilenet_v1_0.75_128 \
mobilenet_v1_0.75_160 \
mobilenet_v1_0.75_192 \
mobilenet_v1_0.75_224 \
mobilenet_v2_0.35_128 \
mobilenet_v2_0.35_160 \
mobilenet_v2_0.35_192 \
mobilenet_v2_0.35_224 \
mobilenet_v2_0.35_96 \
mobilenet_v2_0.5_128 \
mobilenet_v2_0.5_160 \
mobilenet_v2_0.5_96 \
prednet \
resnet18-autoencoder \
resnet18-colorization \
resnet18-contrastive_multiview \
resnet18-contrastive_predictive \
resnet18-deepcluster \
resnet18-depth_prediction \
resnet18-instance_recognition \
resnet18-local_aggregation \
resnet18-relative_position \
resnet18-simclr \
resnet18-supervised \
resnet50-SIN \
resnet50-SIN_IN \
resnet50-SIN_IN_IN \
squeezenet1_0 \
squeezenet1_1; do
	model_list[$i]="$model"
	i=$[$i+1]
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"

python score_models.py "${model_list[$SLURM_ARRAY_TASK_ID]}"