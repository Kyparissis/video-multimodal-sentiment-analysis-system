#!/bin/bash
#SBATCH --job-name=ELECTRA-BASE_finetuning_hidd128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00

python3 ELECTRA-BASE_searchParams_hidd128.py