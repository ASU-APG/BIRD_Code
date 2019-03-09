import sys
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CHANGING THIS TO FIX THE DATA MISMATCH BUG
# DO NOT CHANGE Ebay_dataset.py
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Preparing the color image into dataset class
# Returning:
# (Image, Mask, Color)
# Image and mask would be in the same size, and color will be converted to a one-hot format
		
class EbayColor(Dataset):

	def __init__(self, root_dir, feat_size, arr_code_dict, color_code_dict, transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			feat_size
			# LEGACY from ARRANGEMENT TRAINER
			labels_arr
			labels_col
			images_arr

			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		imag_list = []
		mask_list = []
		imag_dir = os.path.join(root_dir, 'test_images.txt')
		mask_dir = os.path.join(root_dir, 'mask_images.txt')


		# Labels
		labels = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink',
				  'purple', 'red', 'white', 'yellow']
		# Read Image directories
		list_file = open(imag_dir, 'r')
		for i in list_file.readlines():
			imag_list.append(os.path.join(root_dir, i[0:-1].replace('./', '')))


		# Read Mask directories
		list_file = open(mask_dir, 'r')
		for i in list_file.readlines():
			mask_list.append(os.path.join(root_dir, i[0:-1].replace('./', '')))

		self.img_files = imag_list
		self.mask_files = mask_list
		self.length = len(self.img_files)
		self.root_dir = root_dir
		self.transform = transform
		self.feat_size = feat_size
		self.labels = labels
		self.arr_code_dict = arr_code_dict
		self.color_code_dict = color_code_dict
		self.totensor_transform = transforms.Compose([transforms.ToTensor()])


	def __len__(self):
		return self.length

	def __getitem__(self, index):
		imag_direc = self.img_files[index]
		# print("index", index)
		# print(imag_direc)
		# sys.stdout.flush()
		mask_direc = self.mask_files[index]

		image_before_resize = Image.open(imag_direc)
		image = image_before_resize.convert('RGB')
		mask = Image.open(mask_direc)

		if self.transform is not None:
			image = self.transform(image)
			mask = self.transform(mask)

		# Parse out the color label
		color = imag_direc.split('/')[-2]

		# Building up the one-hot label
		label_index = self.labels.index(color)
		one_hot = np.zeros(len(self.labels))
		one_hot[label_index] = 1

		# !!!!!! FROM MBD !!!!!!!
		labels_arr = np.zeros((1, 19))
		labels_col = np.zeros((1, 15))
		num = 0

		imag_fnames = imag_direc.split('/')[-1]
		# print(imag_fnames)

		# for item in imag_fnames:
		# 	print(self.arr_code_dict[item])
		# 	sys.stdout.flush()
		# 	labels_arr[num, self.arr_code_dict[item]] = 1
		# 	num = num+1
		# arrangement = self.labels_arr
		labels_arr[0, self.arr_code_dict[imag_fnames]] = 1
		arrangement = labels_arr

		labels_col[0, :] = self.color_code_dict[imag_fnames]
		color = labels_col

		image_arr = np.float32(np.array(Image.open(imag_direc))) /255.0
		# image_arr = np.transpose(image_arr, (0, 2, 1, 3))
		
		
		# print(color.shape)
		# sample is ---EVERYTHING--- that the arrangement training needs
		# color training will use the other variables returned
		sample = {'image_arr': self.totensor_transform(image_before_resize), 
				  'arrangement': arrangement, 
				  'color': color}

		# print("--------------DIMENSIONS ----------------")
		# print(image.shape, mask.shape, one_hot.shape, 
		# 	  sample['image_arr'].shape, sample['arrangement'].shape, sample['color'].shape)

		return image, mask, one_hot, label_index, sample