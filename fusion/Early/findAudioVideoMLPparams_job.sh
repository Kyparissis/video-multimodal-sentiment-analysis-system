#!/bin/bash
#SBATCH --job-name=findAudioVideoMLPparams
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00

source ../my_enviroment/bin/activate
python3 findAudioVideoMLPparams.py