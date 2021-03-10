#!/bin/bash
#SBATCH --job-name=neural-array
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=150G

i=0
for model in alexnet densenet169 CORnet-Z; do
	model_list[$i]="$model"
	i=$[$i+1]
done

#to output the number of iterations for line 4
echo ${#model_list[@]}
exit