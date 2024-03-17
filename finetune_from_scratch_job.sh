#!/bin/bash
#SBATCH --job-name=finetune_sd1.5_scratch
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --nodelist=xgpe9
#SBATCH --time=120:00:00
#SBATCH --output=finetune1-5-scratch.log

cd /home/y/yuwang/deep_learning2/mobile-diffusion
srun bash finetune_from_scratch.sh
