#!/bin/bash
#SBATCH --job-name=DEBERTA-BASE_finetuning_hidd128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00

python3 DEBERTA-BASE_searchParams_hidd128.py