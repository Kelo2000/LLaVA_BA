#!/bin/bash 

#SBATCH -p nvidia
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:2
#SBATCH --time=80:59:59
#SBATCH -C 80g 
#SBATCH --mem=150GB
#SBATCH -o train_llava_covla.%J.out
#SBATCH -e train_llava_covla.%J.err


#SBATCH --job-name=train

# Environment setup#SBATCH -q ebrainccs
module purge


# Load Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava


./scripts/v1_5/finetune_task.sh