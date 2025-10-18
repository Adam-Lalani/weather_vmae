#!/bin/bash

# Simple script to run three VideoMAE experiments in succession

cd /Users/User/Desktop/Aeolus/weather_mae/vmae_3d_naive

echo "Starting 1h resolution experiment..."
python main.py --temporal_resolution 1h --num_frames 8 --date_start 2021-01-01 --date_end 2021-03-01

echo "Starting 6h resolution experiment..."
python main.py --temporal_resolution 6h --num_frames 8 --date_start 2021-01-01 --date_end 2022-05-16

echo "Starting 12h resolution experiment..."
python main.py --temporal_resolution 12h --num_frames 8 --date_start 2021-01-01 --date_end 2023-09-28

echo "All experiments completed!"
