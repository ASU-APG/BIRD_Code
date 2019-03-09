#!/bin/bash

#SBATCH -N 1                        # number of compute nodes
#SBATCH -n 4                        # number of CPU cores to reserve on this compute node

#SBATCH -p cidsegpu1              # Uncomment this line to use the cidsegpu1 partition instead
#SBATCH -q cidsegpu1                 # Run job under wildfire QOS queue

#SBATCH --gres=gpu:1                # Request two GPUs
#SBATCH -t 1-12:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

module load tensorflow/1.8-agave-gpu

cd /home/tgokhale/work/code/Arrangement_Classification

# pip3 install torch torchvision --user
# pip install tensorboardX --user 
python3 train.py
# python3 test.py 


