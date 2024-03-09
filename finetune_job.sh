#!/bin/bash
#SBATCH --job-name=finetune_sd1.5
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:titanrtx:1        
#SBATCH --nodelist=xgpe4       
#SBATCH --time=24:00:00             
#SBATCH --output=finetune1-5.log   

cd /Users/davidw/Desktop/David/NUS/_Classes/CS5260_deep_learning_2/project/mobile-diffusion/wandb_demo_setup
srun python q5_train_subclass.py