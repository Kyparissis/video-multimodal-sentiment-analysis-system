#!/bin/bash
#SBATCH --job-name=ALBERT-BASE_finetuning_hidd32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00

python3 ALBERT-BASE_searchParams_hidd32.py