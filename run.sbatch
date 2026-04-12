#!/bin/bash
#SBATCH --job-name=collalgo_bench
#SBATCH --account=IscrC_FOCAL
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=log
#SBATCH --error=err

srun ./build/bin/bench
