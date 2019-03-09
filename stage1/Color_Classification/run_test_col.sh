#!/bin/bash

#SBATCH -N 1                        	# number of compute nodes
#SBATCH -n 4                        	# number of CPU cores to reserve on this compute node

#SBATCH -p cidsegpu1              		
#SBATCH -q wildfire                 	# Run job under wildfire QOS queue

#SBATCH --gres=gpu:1                	# Request two GPUs
#SBATCH -t 1-12:00                  	# wall time (D-HH:MM)
#SBATCH -o ./slurm/hardcode_alpha10_beta5.out     	# STDOUT (%j = JobId)
#SBATCH -e ./slurm/slurm.%j.err     	# STDERR (%j = JobId)

module load tensorflow/1.8-agave-gpu

cd /home/tgokhale/work/code/Color_Classification

# pip3 install torch torchvision --user
# pip install tensorboardX --user 
python3 test_hardcode.py




