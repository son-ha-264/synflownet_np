#!/bin/bash
 
#========[ + + + + Requirements + + + + ]========#
#SBATCH --nodelist=isis-001
#SBATCH -o slurm_logs/train_logs_npscore/stdout-%A.%a.output
#SBATCH -e slurm_logs/train_logs_npscore/stderr-%A.%a.error
#SBATCH -n 2 #number of task
#SBATCH -c 2 #number of cpu per task
#SBATCH -J npscore_train_pnpgen
#SBATCH --time=2-00:00:00

 
#========[ + + + + Job Steps + + + + ]========#
source /media/scratch/sonha/.bashrc
conda_initialize

micromamba activate sfn

export CUDA_VISIBLE_DEVICES=1

python /media/scratch/sonha/projects/synflownet_np/src/synflownet/tasks/reactions_task.py