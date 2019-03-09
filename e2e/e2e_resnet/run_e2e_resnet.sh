#!/bin/bash

#SBATCH -N 1                        # number of compute nodes
#SBATCH -n 4                        # number of CPU cores to reserve on this compute node

#SBATCH -p cidsegpu1              # Uncomment this line to use the cidsegpu1 partition instead
#SBATCH -q cidsegpu1              # Run job under wildfire QOS queue

#SBATCH --gres=gpu:1               # Request x GPUs (max 4)
#SBATCH -t 3-12:00                  # wall time (D-HH:MM)

#SBATCH -o slurm_new_5.out              # STDOUT (%j = JobId)
#SBATCH -e slurm_new_5.err	           # STDERR (%j = JobId)

module load tensorflow/1.8-agave-gpu
source ~/work/code/pytorch1_0/bin/activate
# pip3 install pandas
# pip3 install h5py 
cd /home/tgokhale/work/code/e2e_resnet
# python3 csv_h5py_dataloader.py
python3 test.py --learning_rate 0.005 --loss_type mse --batch_size 64 --val_batch_size 64

