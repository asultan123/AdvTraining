#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:00
#SBATCH --job-name=eval_robust
#SBATCH --mem=8GB
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_reports/eval.%j.out
#SBATCH --error=./slurm_reports/eval.%j.err
#SBATCH --partition=ce-mri

echo evaluating $1 and placing results in $2
/home/sultan.a/.pyenv/versions/adv_train/bin/python whitebox.py --model-path $1 --eval-path $2