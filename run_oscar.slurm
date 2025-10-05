#!/bin/bash

#SBATCH -p gpu                # Specify the 'gpu' partition
#SBATCH -N 1
#SBATCH --gres=gpu:2          # Request 2 GPUs for DataParallel training
#SBATCH -n 1                  # Number of tasks (almost always 1 for single-node jobs)
#SBATCH -c 8                  # Number of CPUs per task (2x num_workers for data loading)
#SBATCH --mem=64G             # Request 64GB of memory for large weather datasets
#SBATCH -t 48:00:00           # Set a time limit of 48 hours (weather MAE training can be long)
#SBATCH -J weather-mae        # A descriptive name for your job
#SBATCH -o slurm-%j_mae1.out       # Name of the file to redirect standard output
#SBATCH -e slurm-%j_mae1.err       # Name of the file to redirect standard error

# --- Environment Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Loading required modules..."

module load anaconda
module load cuda

# --- Job Execution ---
echo "Navigating to submission directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

echo "Starting Weather MAE pre-training..."
echo "Dataset: ERA5 weather data"
echo "Model: VideoMAE with 75% masking ratio"
echo "GPUs: 2 (DataParallel)"

/users/allalani/miniconda3/envs/csci1470/bin/python -u main.py

echo "Job finished at $(date)"