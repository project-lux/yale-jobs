#!/bin/bash

#SBATCH --job-name=job
#SBATCH --output=job.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=5
#SBATCH --gres=gpu:h200:1       # Request 1 H200 GPU
#SBATCH --partition=gpu_h200 
#SBATCH --time=48:00:00

conda activate env_name
python script.py