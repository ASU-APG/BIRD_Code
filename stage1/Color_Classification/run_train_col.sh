#!/bin/bash

#SBATCH -N 2                        	# number of compute nodes
#SBATCH -n 4                        	# number of CPU cores to reserve on this compute node

#SBATCH -p cidsegpu1              		
#SBATCH -q wildfire                 	# Run job under wildfire QOS queue

#SBATCH --gres=gpu:1                	# Request two GPUs
#SBATCH -t 2-12:00                  	# wall time (D-HH:MM)
#SBATCH -o ./slurm/slurm.%j.out     	# STDOUT (%j = JobId)
#SBATCH -e ./slurm/slurm.%j.err     	# STDERR (%j = JobId)
#SBATCH --mail-type=ALL             	# Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=tgokhale@asu.edu 	# send-to address

module load tensorflow/1.8-agave-gpu

cd /home/tgokhale/work/code/Color_Classification

# pip3 install torch torchvision --user
# pip install tensorboardX --user 
python3 train.py --checkpoint_path ./checkpoint/lr05_b4_alpha1_beta5 --lr 0.05 --batch_size 4 --alpha 0.1 --beta 0.5

# alpha beta
# -----------
# 0.01	0.1
# 0.01	0.2
# 0.01	0.5
# 0.02	0.1
# 0.02	0.2
# 0.02	0.5 
# 0.05	0.1
# 0.05	0.2
# 0.05	0.5




