#!/bin/bash
#SBATCH --job-name infer-prompt
#SBATCH --output logs/out/%j.txt
#SBATCH --error logs/err/%j.txt
#SBATCH --mail-type ALL
#SBATCH --time 5:00:00
#SBATCH --cpus-per-task 8
#SBATCH --partition allgroups
#SBATCH --mem 10G
#SBATCH --gres=gpu:a40

source .env

srun --mail-user "$EMAIL" ~/.local/bin/uv run python infer.py
