#!/bin/bash
#SBATCH --job-name=VideoMAE_searchParams_hidd32
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=6-00:00:00

source ../../my_enviroment/bin/activate
python3 VideoMAE_searchParams_hidd32.py