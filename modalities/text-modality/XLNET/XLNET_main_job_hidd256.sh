#!/bin/bash
#SBATCH --job-name=XLNET-BASE_finetuning_hidd256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00

python3 XLNET-BASE_searchParams_hidd256.py