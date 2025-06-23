#!/bin/bash

#SBATCH --job-name=serve_models
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus=6            # You might adjust this depending on your cluster
#SBATCH --mem=32G
#SBATCH --output=/mnt/gemini/data1/yifengliu/qe-lr/logs/temp.out
#SBATCH --error=/mnt/gemini/data1/yifengliu/qe-lr/logs/temp.err
#SBATCH --partition=gemini

# project settings
echo "CUDA gpus: $CUDA_VISIBLE_DEVICES"
srun --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32GB --gpus=1 --partition=taurus --account=yifengliu --pty bash -i
