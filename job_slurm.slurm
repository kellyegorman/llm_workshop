#!/bin/bash
#SBATCH --job-name=llm_xxh584
#SBATCH -o llm.o
#SBATCH -A llm_workshop2024
#SBATCH -p aisc_short
#SBATCH --reservation=llm24
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=0:05:00    # Days-HH:MM:SS

singularity exec --nv --bind /mnt/pan/courses/llm24/xxh584:/mnt/pan/courses/llm24/xxh584 /mnt/pan/courses/llm24/xxh584/llm_ws_0.2.sif python 
/Users/kellyg/llm_workshop/test_llm_ws_minimal.py
