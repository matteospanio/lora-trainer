#!/bin/bash
#SBATCH --job-name lora-mistral
#SBATCH --output logs/out/%j.txt
#SBATCH --error logs/err/%j.txt
#SBATCH --mail-type ALL
#SBATCH --time 10-20:00:00
#SBATCH --cpus-per-task 8
#SBATCH --partition allgroups
#SBATCH --mem 20G
#SBATCH --gres=gpu:a40

source .env

srun --mail-user "$EMAIL" ~/.local/bin/uv run python train.py
