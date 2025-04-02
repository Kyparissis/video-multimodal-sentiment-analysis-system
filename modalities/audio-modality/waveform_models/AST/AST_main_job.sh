#!/bin/bash
#SBATCH --job-name=AST_searchParams
#SBATCH --partition=ampere
#SBATCH --qos=ampere-extd
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00

source ../../my_enviroment/bin/activate
python3 AST_searchParams.py