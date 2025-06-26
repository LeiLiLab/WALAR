#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4            # You might adjust this depending on your cluster
#SBATCH --mem=32G
#SBATCH --output=/mnt/gemini/data1/yifengliu/qe-lr/logs/temp.out
#SBATCH --error=/mnt/gemini/data1/yifengliu/qe-lr/logs/temp.err
#SBATCH --partition=taurus

# project settings
echo "CUDA gpus: $CUDA_VISIBLE_DEVICES"

sleep infinity
