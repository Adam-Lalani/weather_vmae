#!/bin/bash

#SBATCH -p gpu                # Specify the 'gpu' partition
#SBATCH -N 1
#SBATCH --gres=gpu:2          # Request 2 GPUs for DataParallel training
#SBATCH -n 1                  # Number of tasks (almost always 1 for single-node jobs)
#SBATCH -c 8                  # Number of CPUs per task (2x num_workers for data loading)
#SBATCH --mem=64G             # Request 32GB of memory. Increase if you get memory errors.
#SBATCH -t 24:00:00           # Set a time limit of 48 hours (HH:MM:SS)
#SBATCH -J vmae-large-1h-6h-12h    # VideoMAE Large experiments with different temporal resolutions
#SBATCH -o slurm-%j.out       # Name of the file to redirect standard output
#SBATCH -e slurm-%j.err       # Name of the file to redirect standard error

# --- Environment Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Loading required modules..."

module load anaconda
module load cuda

# --- Job Execution ---
echo "Navigating to submission directory: $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

echo "Starting VideoMAE Large experiments with different temporal resolutions..."

# 1-hour resolution (8 frames = 8 hours)
echo "--- Running 1-hour resolution experiment ---"
/users/allalani/miniconda3/envs/csci1470/bin/python main.py --temporal_resolution 1h --num_frames 8 --date_start 2021-01-01 --date_end 2021-03-01 --epochs 2000 --batch_size 8 --lr 1e-4

# echo "--- 1-hour resolution experiment finished. Starting 6-hour resolution experiment in 30 seconds ---"
# sleep 30

# # 6-hour resolution (8 frames = 48 hours)
# echo "--- Running 6-hour resolution experiment ---"
# /users/allalani/miniconda3/envs/csci1470/bin/python main.py --temporal_resolution 6h --num_frames 8 --date_start 2021-01-01 --date_end 2022-05-16 --epochs 2000 --batch_size 8 --lr 1e-4

# echo "--- 6-hour resolution experiment finished. Starting 12-hour resolution experiment in 30 seconds ---"
# sleep 30

# # 12-hour resolution (8 frames = 96 hours)
# echo "--- Running 12-hour resolution experiment ---"
# /users/allalani/miniconda3/envs/csci1470/bin/python main.py --temporal_resolution 12h --num_frames 8 --date_start 2021-01-01 --date_end 2023-09-28 --epochs 2000 --batch_size 8 --lr 1e-4

# echo "Job finished at $(date)"
