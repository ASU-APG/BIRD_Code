# BIRD_Code

Please download the data from https://drive.google.com/file/d/1tuwPCe5W3_hI2bWQoSjD15ZU2X0fROSo/view?usp=sharing and unzip it. It should create ./data/ 

- Stage-I     : encoders (Arrangement, Color)

- Stage-II    : fully-connected, q-learning, ILP

- End-to-End  : Resnet, Relational Networks, PSPNet

## Compilation Example
- ```cd ./e2e/e2e_resnet/```
- If you are using SLURM/SBATCH, use ```sbatch run_e2e_resnet.sh```
- For training use ```python3 train.py --learning_rate 0.005 --loss_type mse --batch_size 64 --val_batch_size 64```
- For testing use ```python3 test.py --learning_rate 0.005 --loss_type mse --batch_size 64 --val_batch_size 64```
