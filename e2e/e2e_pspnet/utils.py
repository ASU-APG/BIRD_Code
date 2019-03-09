import csv
import pandas as pd
import numpy as np 
from PIL import Image 
import sys

from torch.utils.data import Dataset, DataLoader

def read_batch(csv_rows, batchnum, batchsize):
	skiprows = batchnum * batchsize

	x = np.zeros((batchsize, 256, 256, 6))
	y = np.zeros((batchsize, 128))

	for i in range(batchsize):

		source = np.expand_dims(
					np.float32(
						np.array(
							Image.open(
								'../../data/all_256_256/'
								+ csv_rows[skiprows+i][0] 
								+ '.jpg')))/255.0, 
					axis=0)

		target = np.expand_dims(
					np.float32(
						np.array(
							Image.open(
								'../../data/all_256_256/' 
								+ csv_rows[skiprows+i][1] 
								+ '.jpg')))/255.0,
					axis=0)

		x[i, :, :, :] = np.concatenate((source, target), axis=3)

		label = csv_rows[skiprows+i][2]
		if label == 0:
			y[i, :] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		else:
			y[i, :] = [int(x) for x in list(label)]
		
		
	x = np.transpose(x, (0, 3, 1, 2))
	return x, y

def read_csv_batch(filename, batchnum, batchsize):
	csv_rows = pd.read_csv(filename, skiprows=batchnum*batchsize, nrows=batchsize, header=None)
	dropcols = [1, 3]
	csv_rows = csv_rows.drop(csv_rows.columns[dropcols], axis=1)
	csv_rows = csv_rows.values.tolist()

	x = np.zeros((batchsize, 256, 256, 6))
	y = np.zeros((batchsize, 128))

	for i in range(len(csv_rows)):
		# print(i)
		sys.stdout.flush()

		source = np.expand_dims(
					np.float32(
						np.array(
							Image.open(
								'../../data/all_256_256/' + csv_rows[i][0] + '.jpg')))/255.0, 
					axis=0)

		target = np.expand_dims(
					np.float32(
						np.array(
							Image.open(
								'../../data/all_256_256/' + csv_rows[i][1] + '.jpg')))/255.0,
					axis=0)

		x[i, :, :, :] = np.concatenate((source, target), axis=3)

		label = csv_rows[i][3]
		if label == 0:
			y[i, :] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		else:
			y[i, :] = [int(x) for x in list(label)]
		
		
	x = np.transpose(x, (0, 3, 1, 2))
	return x, y

class CSVDataset(Dataset):
	def __init__(self, path, batchsize, nb_samples):
		self.path = path 
		self.batchsize = batchsize 
		self.len = int(nb_samples / self.batchsize)

	def __getitem__(self, index):
		csv_rows = pd.read_csv(self.path, skiprows=index*self.batchsize, nrows=self.batchsize, header=None)
		# dropcols = [1, 3]
		# csv_rows = csv_rows.drop(csv_rows.columns[dropcols], axis=1)
		csv_rows = csv_rows.values.tolist()

		x = np.zeros((self.batchsize, 256, 256, 6))
		y = np.zeros((self.batchsize, 48))

		for i in range(self.batchsize):
			# print(i)
			sys.stdout.flush()

			source = np.expand_dims(
						np.float32(
							np.array(
								Image.open(
									'../../data/all_256_256/' + csv_rows[i][0] + '.jpg')))/255.0, 
						axis=0)

			target = np.expand_dims(
						np.float32(
							np.array(
								Image.open(
									'../../data/all_256_256/' + csv_rows[i][2] + '.jpg')))/255.0,
						axis=0)

			x[i, :, :, :] = np.concatenate((source, target), axis=3)

			label = csv_rows[i][5]
			if label == 0:
				y[i, :] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			else:
				y[i, :] = [int(x) for x in list(label)]
			
			
		x = np.transpose(x, (0, 3, 1, 2))

		print(x.shape, y.shape)
		sys.stdout.flush()

		sample = {'x': x, 'y': y}
		return sample

	def __len__(self):
		return self.len 



