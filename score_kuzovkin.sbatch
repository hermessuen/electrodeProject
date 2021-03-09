#!/bin/bash
#SBATCH --job-name=neural-array
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=150G




python score_models.py 