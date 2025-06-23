#!/usr/bin/env bash

# Specifying every sbatch parameters will make things easier

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gpus=1
##SBATCH --constraint=xeon-4116 (some node property to request)
#SBATCH --partition=aries
#SBATCH --time=1-2:34:56 (1 day 2 hour 34 min 56 sec)
#SBATCH --dependency=afterok:job_id
#SBATCH --array=1-3 ($SLURM_ARRAY_TASK_ID)
#SBATCH --account=yifengliu
#SBATCH --mail-type=begin|end|fail|all (Email notification)
#SBATCH --mail-user=lyfeng0702@gmail.com
#SBATCH --output=/mnt/gemini/data1/yifengliu/qe-lr/outputstdout.txt
#SBATCH --error=/mnt/gemini/data1/yifengliu/qe-lr/outputstderr.txt

# The rest are your jobs

## Use environment from gemini
conda config --append envs_dirs /mnt/gemini/data1/yifengliu/miniconda3
source /mnt/gemini/data1/yifengliu/miniconda3/bin/activate qe-rl

## Run your job
bash /mnt/gemini/data1/yifengliu/qe-lr/scripts/predict.sh
