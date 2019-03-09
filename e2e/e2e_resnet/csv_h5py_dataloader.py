import sys 
import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image 

chunksize = 2000
num_lines = 600000

csv_path = '../../data/final_plans/final_train.csv'

with h5py.File('./final_train_cpu.h5', 'w') as h5f:
	dset1 = h5f.create_dataset('images', shape=(num_lines, 6, 256, 256), compression=None)
	dset2 = h5f.create_dataset('labels', shape=(num_lines, 48), compression=None)

	for i in range(0, num_lines, chunksize):
		print(i)
		sys.stdout.flush()
		csv_rows = pd.read_csv(csv_path, header=None, nrows=chunksize, skiprows=i)
		csv_rows = csv_rows.values.tolist()

		x = np.zeros((chunksize, 256, 256, 6))
		y = np.zeros((chunksize, 48))

		for j in range(chunksize):
			source = np.expand_dims(
						np.float32(
							np.array(
								Image.open(
									'../../data/all_256_256/'
									+ csv_rows[j][0]
									+ '.jpg'))) /255.0,
						axis=0
						)

			target = np.expand_dims(
						np.float32(
							np.array(
								Image.open(
									'../../data/all_256_256/'
									+ csv_rows[j][2]
									+ '.jpg'))) /255.0,
						axis=0
						)

			x[j,:,:,:] = np.concatenate((source, target), axis=3)

			label = csv_rows[j][5]
			if label == 0:
				y[j, :] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			else:
				y[j, :] = [int(x) for x in list(label)]
			
			
		x = np.transpose(x, (0, 3, 1, 2))

		dset1[i:i+chunksize, :, :, :] = x
		dset2[i:i+chunksize, :] = y

with h5py.File('./final_train.h5', 'r') as h5f:
    print(h5f['images'].shape)
    print(h5f['labels'].shape)
