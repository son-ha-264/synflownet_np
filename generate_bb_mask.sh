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

#python /media/scratch/sonha/projects/synflownet_np/src/synflownet/data/building_blocks/precompute_bb_masks.py \
#--add_hs \
#--building_blocks /media/scratch/sonha/projects/synflownet_np/src/synflownet/data/building_blocks/synthemol_bbs.txt \
#--templates /media/scratch/sonha/projects/synflownet_np/src/synflownet/data/templates/synthemol_reaction.txt

python /media/scratch/sonha/projects/synflownet_np/logs/debug_run_reactions_task_2026-11-09_12-26-07/inspect_batch.py /media/scratch/sonha/projects/synflownet_np/logs/debug_run_reactions_task_2026-11-09_12-26-07/dump.pkl