#!/bin/bash
#SBATCH --job-name=HUBERT_searchParams
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00

source ../../my_enviroment/bin/activate
python3 HUBERT_searchParams.py