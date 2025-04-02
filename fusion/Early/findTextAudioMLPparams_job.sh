#!/bin/bash
#SBATCH --job-name=findTextAudioMLPparams
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00

source ../my_enviroment/bin/activate
python3 findTextAudioMLPparams.py