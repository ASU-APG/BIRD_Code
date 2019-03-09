	
from __future__ import division, print_function, absolute_import
from PIL import Image 
import glob 
import csv 
import numpy as np 

def get_file_list(train_path, test_path, eval_path):
	""" 
	returns the list of all image filenames in train, test, eval subsets
	"""

	train_list = glob.glob(train_path + '*.jpg')
	test_list = glob.glob(test_path + '*.jpg')
	eval_list = glob.glob(eval_path + '*.jpg')

	return train_list, test_list, eval_list

def load_img_data(train_list, test_list, eval_list):
	"""
	loads images as [#imgs, height, width, channels] format
	"""

	train_data = np.float32(np.array([np.array(Image.open(fname)) for fname in train_list]))/255.0
	test_data = np.float32(np.array([np.array(Image.open(fname)) for fname in test_list]))/255.0
	eval_data = np.float32(np.array([np.array(Image.open(fname)) for fname in eval_list]))/255.0

	return train_data, test_data, eval_data

def load_labels_arr(train_list, test_list, eval_list):

	train_flist = []
	for item in train_list:
		train_flist.append(item.split('/')[-1])

	test_flist = []
	for item in test_list:
		test_flist.append(item.split('/')[-1])

	eval_flist = []
	for item in eval_list:
		eval_flist.append(item.split('/')[-1])

	arr_code_dict ={}
	with open('../../data/arrangement_class.csv') as f:
	  reader = csv.reader(f)
	  for row in reader:
	    key = row[0]
	    arr_code_dict[key] = int(row[-1])

	train_labels_arr = np.zeros((len(train_flist), 19))
	num = 0
	for item in train_flist:
	  train_labels_arr[num, arr_code_dict[item]] = 1
	  num = num+1

	test_labels_arr = np.zeros((len(test_flist), 19))
	num = 0
	for item in test_flist:
	  test_labels_arr[num, arr_code_dict[item]] = 1
	  num = num+1

	eval_labels_arr = np.zeros((len(eval_flist), 19))
	num = 0
	for item in eval_flist:
	  eval_labels_arr[num, arr_code_dict[item]] = 1
	  num = num+1

	return train_labels_arr ,test_labels_arr, eval_labels_arr


def load_labels_color(train_list, test_list, eval_list):
	# read and store labels (i.e. color_codes)
	train_flist = []
	for item in train_list:
		train_flist.append(item.split('/')[-1])

	test_flist = []
	for item in test_list:
		test_flist.append(item.split('/')[-1])

	eval_flist = []
	for item in eval_list:
		eval_flist.append(item.split('/')[-1])

	color_code_dict = {}
	with open('../../data/color_code.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			key = row[0]
			rest = row[1:]
			ccode = [float(i) for i in rest]
			color_code_dict[key] = np.array(ccode)


	train_labels_color = np.zeros((len(train_flist), 15))
	num = 0
	for item in train_flist:
		train_labels_color[num, :] = color_code_dict[item]
		num = num+1

	test_labels_color = np.zeros((len(test_flist), 15))
	num = 0
	for item in test_flist:
		test_labels_color[num, :] = color_code_dict[item]
		num = num+1

	eval_labels_color = np.zeros((len(eval_flist), 15))
	num = 0
	for item in eval_flist:
		eval_labels_color[num, :] = color_code_dict[item]
		num = num+1

	return train_labels_color, test_labels_color, eval_labels_color 



def readcsv(filename):  
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=",")

    rownum = 0  
    a = []

    for row in reader:
        a.append (row)
        rownum += 1
    
    ifile.close()
    return a
