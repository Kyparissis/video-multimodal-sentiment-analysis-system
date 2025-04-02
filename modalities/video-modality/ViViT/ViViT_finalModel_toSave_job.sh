#!/bin/bash
#SBATCH --job-name=ViViT_finalModel_toSave
#SBATCH --partition=ampere
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=2:59:00

source ../../my_enviroment/bin/activate
python3 ViViT_finalModel_toSave.py