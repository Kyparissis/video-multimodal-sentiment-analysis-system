#!/bin/bash
#SBATCH --job-name=Wav2VecBASE_searchParams
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=2-12:00:00

source ../../my_enviroment/bin/activate
python3 Wav2Vec_searchParams.py