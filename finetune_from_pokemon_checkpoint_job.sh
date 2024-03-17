#!/bin/bash
#SBATCH --job-name=finetune_sd1.5_continue_from_ckpt
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --nodelist=xgpe4
#SBATCH --time=120:00:00
#SBATCH --output=finetune1-5-cont.log

cd /home/y/yuwang/deep_learning2/mobile-diffusion
srun bash finetune_from_pokemon_checkpoint.sh
