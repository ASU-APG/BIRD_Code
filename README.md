# Blocksworld Revisited + Blocksworld Image Reasoning Dataset

Please download the data from https://drive.google.com/file/d/1tuwPCe5W3_hI2bWQoSjD15ZU2X0fROSo/view?usp=sharing and unzip it. It should create ./data/ 

- Stage-I     : encoders (Arrangement, Color)

- Stage-II    : fully-connected, q-learning, ILP

- End-to-End  : Resnet, Relational Networks, PSPNet

## Dataset Features
Inside ```./data``` you'll find the data needed to run our experiments. This is v0.1 of the Blocksworld Image Reasoning Dataset.  
### Images
- ```all_256_256``` contains all 7267 blocksworld images, whie train-test-val splits can be found in ```train_256_256```, ```test_256_256``` and ```eval_256_256```

### Event-Sequence Data
- CSVs required to run End-to-End experiments can be found in ```./data/final_plans/```. This includes datasets created for our baselines as well as ablation studies. This data contains sequences as 128bit binary vectors
- CSVs for Stage-II experiments can be found in ./data/gt_plans . This data contains sequences as textual descriptions of the form mov(X, Y, t) as explained in the paper.
## Compilation Example
- ```cd ./e2e/e2e_resnet/```
- If you are using SLURM/SBATCH, submit batch scripts using ```sbatch run_e2e_resnet.sh```
- For training run```python3 train.py --learning_rate 0.005 --loss_type mse --batch_size 64 --val_batch_size 64```
- For testing run ```python3 test.py --learning_rate 0.005 --loss_type mse --batch_size 64 --val_batch_size 64```


