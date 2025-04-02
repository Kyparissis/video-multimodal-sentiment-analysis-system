#!/bin/bash
#SBATCH --job-name=ROBERTA-LARGE_finetuning_hidd64-01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=23:59:00

python3 ROBERTA-LARGE_searchParams_hidd64-01.py