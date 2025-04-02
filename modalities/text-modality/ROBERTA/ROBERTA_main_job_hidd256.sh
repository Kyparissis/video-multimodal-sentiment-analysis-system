#!/bin/bash
#SBATCH --job-name=ROBERTA-BASE_finetuning_hidd256
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00

python3 ROBERTA-BASE_searchParams_hidd256.py