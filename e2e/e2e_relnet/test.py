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
from model import RN

# Training settings
parser = argparse.ArgumentParser(description='e2e_resnet')
parser.add_argument('--learning_rate', type=float, default=0.001, help='lr (default: 0.0001)')
parser.add_argument('--num_epochs', type=int, default=200, help='num training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch-size')
parser.add_argument('--val_batch_size', type=int, default=64, help='val-batch-size')
parser.add_argument('--num_train_samples', type=int, default=600000, help='num_train_samples')
parser.add_argument('--num_val_samples', type=int, default=20000, help='num_val_samples')
parser.add_argument('--loss_type', type=str, default='mse', help='type of loss to use')
parser.add_argument('--train_path', type=str, default='../../data/final_plans/final_train_1hot.csv', help='loc of trian csv')
parser.add_argument('--val_path', type=str, default='../../data/final_plans/final_val_1hot.csv', help='loc of val csv')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

if __name__ == '__main__':

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	gpu_num = torch.cuda.device_count()
	print('GPU NUM: {:2d}'.format(gpu_num))

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

	# data_train = pd.read_csv(train_path, 
	# 						skiprows=0,
	# 						nrows=num_train_samples,
	# 						header=0,
	# 						usecols=[0,2,5])
	# rows_train = data_train.values.tolist()

	data_val = pd.read_csv(val_path, 
							skiprows=0,
							nrows=num_val_samples,
							header=0,
							usecols=[0,2,5])
	rows_val = data_val.values.tolist()


	# MODEL
	# -----------------------------
	model = RN()
	if torch.cuda.device_count() > 1:
	  print("Let's use", torch.cuda.device_count(), "GPUs!")
	  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
	  model = nn.DataParallel(model)
	model.to(device)

	best_acc = 0

	# LOSS
	# -----------------------------
	if loss_type== 'bce':
		criterion = nn.BCELoss()
	else:
		criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	print("model and criterion loaded ...")


	
	checkpoint = torch.load('./saved_models/model_arr_20190223_21_0.0001_64_mse.ckpt')
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	epoch = checkpoint['epoch']

	model.eval()
	with torch.no_grad():
		count = 0
		total = 0
		prec_count = 0
		blankz_count = 0
		for i in range(num_val_iters-1):
			# x, y = read_csv_batch('../../data/final_plans/final_val.csv', i, val_batch_size)
			x,y = read_batch(rows_val, i, val_batch_size)
				
			x = torch.tensor(x)
			x = nn.functional.interpolate(x, (75, 75))

			img = x[:, :3, :, :]
			img2 = x[:, 3:, :, :]

			img = img.to(device).float()
			qst = img2.to(device).float()

			y = torch.tensor(y).to(device).float()
			y_rs = y.view(y.size(0), 8, 16)
			zeros_rs = torch.tensor([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1]).to(device)

			# forward pass
			y_pred = model(img, qst).round()
			y_pred_rs = y_pred.view(y_pred.size(0), 8, 16)

			partialz = torch.sum(torch.all(torch.eq(y_rs, y_pred_rs), dim=2)).item()
			blankz = torch.sum(torch.all(torch.eq(y_rs, zeros_rs.float()), dim=2)).item()
			equalz = torch.sum(torch.all(torch.eq(y, y_pred.float()), dim=1)).item()

			count = count + equalz
			blankz_count = blankz_count + blankz
			prec_count = prec_count + (partialz - blankz)/(8*val_batch_size - blankz)
			total = total + val_batch_size
			print("blankz: ", blankz, "partialz:", partialz, "precision now: ", prec_count/(i+1), "accuracy now", count/total)
			sys.stdout.flush()

			small_val_acc = count/total
		small_val_prec = prec_count/num_val_iters
		is_best = (small_val_acc >= best_acc)
		print('Epoch {} : Val Accuracy: {}, Val Precision: {}, is_best: {}'.format(epoch+1, small_val_acc, small_val_prec, is_best)) 
		sys.stdout.flush()