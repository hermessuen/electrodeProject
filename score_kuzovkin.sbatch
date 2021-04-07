#!/bin/bash
#SBATCH --job-name=neural-array
#SBATCH --array=0-33
#SBATCH --time=92:00:00
#SBATCH --ntasks=1
#SBATCH --mem=300000

i=0
for model in alexnet resnet-18 xception densenet-121 densenet-121 densenet-169 densenet-201 inception_v1 inception_v2inception_v3 inception_v4 inception_resnet_v2 resnet-18 resnet-34 resnet-50-pytorch resnet-50_v1 resnet-101_v1 resnet-152_v1 resnet-50_v2 resnet-101_v2 resnet-152_v2 resnet-50-robust vgg-16 vgg-19 vggface bagnet9 bagnet17 bagnet33 nasnet_mobile nasnet_large pnasnet_large mobilenet_v1_1.0_224 mobilenet_v1_1.0_192 mobilenet_v1_1.0_160 mobilenet_v1_1.0_128; do
	model_list[$i]="$model"
	i=$[$i+1]
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model ${model_list[$SLURM_ARRAY_TASK_ID]}"

python score_models.py "${model_list[$SLURM_ARRAY_TASK_ID]}"