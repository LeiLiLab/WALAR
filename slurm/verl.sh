#!/usr/bin/env bash

# Specifying every sbatch parameters will make things easier

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gpus=4
#SBATCH --partition=taurus
#SBATCH --output=/mnt/gemini/data1/yifengliu/qe-lr/outputstdout.txt
#SBATCH --error=/mnt/gemini/data1/yifengliu/qe-lr/outputstderr.txt


eval "$(/mnt/gemini/home/yifengliu/miniconda3/bin/conda shell.bash hook)"
which python
source /mnt/gemini/home/yifengliu/miniconda3/bin/activate qe-rl

bash /mnt/gemini/data1/yifengliu/MT-R1-Zero/main_grpo.sh
