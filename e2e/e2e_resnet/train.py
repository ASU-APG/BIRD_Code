import os
import sys
import csv
import glob
import time
import argparse 
import datetime
import numpy as np 
from random import shuffle

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils import *
from resnet import resnet50


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_num = torch.cuda.device_count()
print('GPU NUM: {:2d}'.format(gpu_num)) 


# Training settings
parser = argparse.ArgumentParser(description='e2e_resnet')
parser.add_argument('--lr', type=float, default=0.001, help='lr (default: 0.0001)')
parser.add_argument('--num_epochs', type=int, default=200, help='num training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch-size')
parser.add_argument('--val_batch_size', type=int, default=64, help='val-batch-size')
parser.add_argument('--num_train_samples', type=int, default=600000, help='num_train_samples')
parser.add_argument('--num_val_samples', type=int, default=200000, help='num_val_samples')
parser.add_argument('--loss_type', type=str, default='mse', help='type of loss to use')
parser.add_argument('--train_path', type=str, default='../../data/final_plans/final_train_1hot.csv', help='loc of trian csv')
parser.add_argument('--val_path', type=str, default='../../data/final_plans/final_val_1hot.csv', help='loc of val csv')


if __name__ == '__main__':
	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	args = parser.parse_args()

	# ARGS
	# -----------------------------
	num_epochs = args.num_epochs
	batch_size = args.batch_size
	val_batch_size = args.val_batch_size
	learning_rate = args.learning_rate
	loss_type = args.loss_type 
	num_train_samples = args.num_train_samples
	num_val_samples = args.num_val_samples
	train_path = args.train_path
	val_path = args.val_path 

	# -----------------------------
	num_iters = int(num_train_samples/batch_size)
	num_val_iters = int(num_val_samples/val_batch_size)

	data_train = pd.read_csv(train_path, 
							skiprows=0,
							nrows=num_train_samples,
							header=0,
							usecols=[0,2,5])
	rows_train = data_train.values.tolist()

	data_val = pd.read_csv(val_path, 
							skiprows=0,
							nrows=num_val_samples,
							header=0,
							usecols=[0,2,5])
	rows_val = data_val.values.tolist()


	# MODEL
	# -----------------------------
	model = resnet50(False)
	if torch.cuda.device_count() > 1:
	  print("Let's use", torch.cuda.device_count(), "GPUs!")
	  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
	  model = nn.DataParallel(model)
	model.to(device)


	# LOSS
	# -----------------------------
	if loss_type== 'bce':
		criterion = nn.BCELoss()
	else:
		criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	print("model and criterion loaded ...")

	# LOOP
	# -----------------------------
	for epoch in range(num_epochs):
		model.train()
		shuffle(rows_train)

		# TRAIN one EPOCH
		# -------------------------
		for i in range(num_iters-1):
			# read data
			# x, y = read_csv_batch('../../data/final_plans/final_train.csv', i, batch_size)
			ts = time.time()
			x,y = read_batch(rows_train, i, batch_size)

			x = torch.tensor(x).to(device).float()
			y = torch.tensor(y).to(device).float()
			# print("time to load data", time.time() - ts)
			# sys.stdout.flush()
			# forward pass
			y_pred = model(x)
			# print("time for fwd pass", time.time() - ts)
			# sys.stdout.flush()
			# update
			loss = criterion(y_pred, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if ((i+1) % (int(num_iters/100) + 1) == 0):
				print("source illustration", rows_train[i*batch_size][0])
				print("target illustration", rows_train[i*batch_size][1])
				print("plan illustration", rows_train[i*batch_size][2])
				print("predicted: ", y_pred[0].data)
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					   .format(epoch+1, num_epochs, i+1, num_iters, loss.item()))
				sys.stdout.flush()

		# SAVE MODEL CHECKPOINT
		# -------------------------
		if not os.path.exists('./saved_models/'):
		  os.makedirs('./saved_models/')
		ts = time.time()
		st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d')
		uniq_id = st +"_" + str(epoch) +"_" + str(learning_rate) +"_" + str(batch_size) + "_"+ loss_type
		torch.save(model.state_dict(), "./saved_models/model_arr_" + uniq_id + ".ckpt")


		# GET VAL ACCURACY
		model.eval()
		with torch.no_grad():
			count = 0
			total = 0
			for i in range(num_val_iters):
				# x, y = read_csv_batch('../../data/final_plans/final_val.csv', i, val_batch_size)
				x,y = read_batch(rows_val, i, val_batch_size)
				x = torch.tensor(x).to(device).float()
				y = torch.tensor(y).to(device)

				# forward pass
				y_pred = model(x).round()

				equalz = torch.sum(torch.all(torch.eq(y, y_pred.double()), dim=1)).item()
				count = count + equalz
				total = total + val_batch_size

			print('Test Accuracy after Epoch {} : {}'.format(epoch+1, count/total))
			sys.stdout.flush()
