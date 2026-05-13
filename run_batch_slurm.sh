#!/bin/bash
#SBATCH --job-name=sqz-crit
#SBATCH --output=logs/sqz-crit_%A_%a.out
#SBATCH --error=logs/sqz-crit_%A_%a.err
#SBATCH --array=1-7  # adjust to match the number of jobs in run_batch.jl
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8

mkdir -p logs

# Load Julia module (adjust for your cluster)
# module load julia/1.12

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

julia --project=. run_batch.jl
