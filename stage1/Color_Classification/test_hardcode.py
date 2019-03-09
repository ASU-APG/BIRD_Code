import os
import random
import torch
import pickle
import numpy as np
from models.resnet import resnet50
from lib.dataset.Ebay_dataset import EbayColor
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
from PIL import Image 
import cv2 
import glob 
import csv 
import sys
import statistics 
from scipy import signal


# Load the dataset
feat_size = (32, 32)
size = (512, 512)

# Color codes
# -----------------
# black, BLUE, brown, GREEN, gray, ORANGE, 
# pink, PURPLE, RED, white, YELLOW
color_dict = [[255,255,255], [0,0,255], [255,255,255], [0,128,128], [255,255,255], [255,165,0],
			  [255,255,255], [128,0,128], [255,0,0], [255,255,255], [255,255,0]]
color_3bit = [[0,0,0], [0,1,1], [0,0,0], [0,1,0], [0,0,0], [1,1,0], 
			  [0,0,0], [1,0,1], [0,0,1], [0,0,0], [1,0,0]]
valid_colors = [1,3,5,7,8,10] # BLUE, GREEN, ORANGE, PURPLE, RED, YELLOW
color_list = []

for i in range(11):
	color_list.append([item/255. for item in color_dict[i]])

# LOAD MODEL FROM CHECKPOINT
# ---------------------------
transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
data_set = EbayColor('./blocks_data/', feat_size, transform)
model = resnet50(False, '', 11)
model.fc = torch.nn.Linear(512, 11)

# Evaluation mode for batch normalization freeze
model.eval()
for p in model.parameters():
	p.requires_grad = False

# Load Back bone Module
state_dict = torch.load('./checkpoint/lr05_b4_alpha10_beta5/Color_pretrain_regression_96.pth')['state_dict']
new_params = model.state_dict()
new_params.update(state_dict)
model.load_state_dict(new_params)
model = model.cuda()

def predict_color(conv_feat3, model):
	result = np.zeros((128, 128, 3))
	colormap = np.zeros((128, 128))
	# Iterating through pixels embeddings
	for i in range(conv_feat3.shape[-2]):
		for j in range(conv_feat3.shape[-1]):
			# Penalize the pixels of interests
			pixel_feat = conv_feat3[0, :, i, j]
			pixel_feat = pixel_feat.contiguous().view(1, -1)
			prediction = model.fc(pixel_feat)
			_, predicted = torch.max(prediction.data, 1) 
			predicted = predicted.cpu().numpy()[0]
			colormap[i,j] = predicted
			result[i,j] = [item/255. for item in color_dict[predicted]]
	
	return result, colormap


def gridify(centroids):
	centroids = np.nan_to_num(centroids)
	return np.ceil(centroids/12.8).astype(int)








# LOAD TEST DATA
# -----------------
test_list = glob.glob('../../data/all_256_256/*.jpg')
test_flist = []
for item in test_list:
	test_flist.append(item.split('/')[-1])

arr_code_dict ={}
with open('../../data/arrangement_class.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		key = row[0]
		arr_code_dict[key] = int(row[-1])
		
test_labels_arr = np.zeros((len(test_flist), 1))
num = 0
for item in test_flist:
	test_labels_arr[num, :] = arr_code_dict[item]
	num = num+1

# print(test_labels_arr.shape)
# print(test_labels_arr[1])
# print(test_labels_arr[1, 0]==17)
color_code_dict = {}
with open('../../data/color_code.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		key = row[0]
		rest = row[1:]
		ccode = [float(i) for i in rest]
		color_code_dict[key] = np.array(ccode)

test_labels_color = np.zeros((len(test_flist), 15))
num = 0
for item in test_flist:
	test_labels_color[num, :] = color_code_dict[item]
	num = num+1

size = (1024, 1024)
transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])

tot_horz = 0
nott = 0
ecc = 0 # exactly correct colors
ccc = 0	# count of common colors
ttt = 0 # total
ttb = 0 # total blocks


for n in range(len(test_list)):
	image_path = test_list[n]

	image = Image.open(image_path)	
	image = transform(image)
	images = Variable(image.view(1,3,1024, 1024)).cuda()
	conv_feat3, conv_feat2 = model(images)
	result, colormap = predict_color(conv_feat2, model)
	plt.imsave('./output/' + test_flist[n][:-4] + '.png', result)

	colormap_medfilt = signal.medfilt(colormap, (7,7))

	medians = np.zeros((6, 2))

	locs_x = {}
	locs_y = {}
	medians_x = {}
	medians_y = {}

	for key in range(6):
		locs_x[key] = []
		locs_y[key] = []


	masks = np.zeros((128, 128, 11))
	

	for color_idx in range(11):
		masks[:, :, color_idx] = (colormap==color_idx)*1

		this_mask = np.array(masks[:, :, color_idx], dtype = np.uint8)

		im, contours, hierarchy = cv2.findContours(this_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )

		maxContour = 0
		flag = 0
		for contour in contours:
			contourSize = cv2.contourArea(contour)
			if contourSize > maxContour:
				flag = flag + 1
				maxContour = contourSize
				maxContourData = contour

		# Create a mask from the largest contour
		mask = np.zeros_like(this_mask)
		mux = np.ones_like(this_mask)
		# print(maxContour)
		if abs(maxContour) > 100:
			cv2.fillPoly(mask,[maxContourData],1)

		# Use mask to crop data from original image
		finalImage = np.zeros_like(this_mask)
		finalImage = np.multiply(mux,mask)

		masks[:, :, color_idx] = finalImage

		if color_idx in valid_colors:
			for i in range(masks.shape[0]):
				for j in range(masks.shape[1]):
					if masks[i, j, color_idx] == 1:
						locs_x[valid_colors.index(color_idx)] = locs_x[valid_colors.index(color_idx)] + [i]
						locs_y[valid_colors.index(color_idx)] = locs_y[valid_colors.index(color_idx)] + [j]
	

	for key in range(6):
		if locs_x[key]:
			medians[key, 0] = statistics.median(locs_x[key])
		if locs_y[key]:
			medians[key, 1] = statistics.median(locs_y[key])


	gridified_medians = gridify(medians)
	gridified_centroids = gridified_medians


	color_vector = [] 


	
	# Case HORZ:	1-1, 1-1-1, 1-1-1-1, 1-1-1-1-1
	if test_labels_arr[n, :] in [6, 12, 16, 18]:
		print("CASE 1HORZ")
		median_sortlist = medians[:, 1].argsort()
		for m in median_sortlist:
			if medians[m, 1] != 0:
				color_vector.append(color_3bit[valid_colors[m]])
		# print(color_vector)
		tot_horz = tot_horz + 1
		ttt = ttt + 1


	
	# Case VERT:	1, 2, 3, 4, 5, 6
	elif test_labels_arr[n, :] in [1, 2, 3, 4, 5]:
		print("CASE 1VERT")
		median_sortlist = medians[:, 0].argsort()[::-1]
		for m in median_sortlist:
			if medians[m, 0] != 0:
				color_vector.append(color_3bit[valid_colors[m]])
		ttt = ttt + 1


	# Case 2a:		1-2, 1-3, 1-4
	elif test_labels_arr[n, :] in [7, 8, 9]:
		print("CASE 2a")
		median_sortlist = medians[:, 1].argsort()
		# get leftmost first
		
		for m in median_sortlist:
			if medians[m, 0] != 0:
				color_vector.append(color_3bit[valid_colors[m]])
				medians[m, :] = 0
				break

		# then arrange the rest vertically
		median_sortlist = medians[:, 0].argsort()[::-1]
		for m in median_sortlist:
			if medians[m, 0] != 0:
				color_vector.append(color_3bit[valid_colors[m]])
		ttt = ttt + 1

	# # Case 2b: 2-2, 2-3
	elif test_labels_arr[n, :] in [10, 11]:
		print("CASE 2b")
		median_sortlist = medians[:, 1].argsort()
		# get leftmost first
		count_left = 0
		for m in median_sortlist:
			if medians[m, 1] != 0:
				if (count_left == 0):
					first_x_index = m
				if (count_left == 1):
					second_x_index = m
				medians[m, :] = 0
				count_left = count_left + 1
				if count_left > 1:
					break

		if (medians[first_x_index,0] > medians[second_x_index,0]):
			color_vector.append(color_3bit[valid_colors[first_x_index]])
			color_vector.append(color_3bit[valid_colors[second_x_index]])
		else:
			color_vector.append(color_3bit[valid_colors[second_x_index]])
			color_vector.append(color_3bit[valid_colors[first_x_index]])		
				
				
			# medians[m, :] = 0

		# then arrange the rest vertically
		median_sortlist = medians[:, 0].argsort()[::-1]
		count_right = 0
		for m in median_sortlist:
			if medians[m, 0] != 0:
				color_vector.append(color_3bit[valid_colors[m]])
				count_right = count_right + 1
				if count_right > 2:
					break
		ttt = ttt + 1

	# Case 3a:		1-1-2, 1-1-3
	elif test_labels_arr[n, :] in [13, 14]:
		print("CASE 3a")
		median_sortlist = medians[:, 1].argsort()
		# get leftmost first
		count_left = 0
		for m in median_sortlist:
			if medians[m, 1] != 0:
				color_vector.append(color_3bit[valid_colors[m]])
				medians[m, :] = 0
				count_left = count_left + 1
				if count_left > 1:
					break
			# medians[m, :] = 0

		# then arrange the rest vertically
		median_sortlist = medians[:, 0].argsort()[::-1]
		count_right = 0
		for m in median_sortlist:
			if medians[m, 0] != 0:
				color_vector.append(color_3bit[valid_colors[m]])
				count_right = count_right + 1
				if count_right > 2:
					break
		ttt = ttt + 1


	# # # Case 3b:		1-2-2
	elif test_labels_arr[n, :] in [15]:
		print("CASE 3b")
		median_sortlist = medians[:, 1].argsort()
		# get leftmost first
		
		for m in median_sortlist:
			if medians[m, 0] != 0:
				color_vector.append(color_3bit[valid_colors[m]])
				medians[m, :] = 0
				break

		count_left = 0
		for m in median_sortlist:
			if medians[m, 1] != 0:
				if (count_left == 0):
					first_x_index = m
				if (count_left == 1):
					second_x_index = m
				medians[m, :] = 0
				count_left = count_left + 1
				if count_left > 1:
					break

		if (medians[first_x_index,0] > medians[second_x_index,0]):
			color_vector.append(color_3bit[valid_colors[first_x_index]])
			color_vector.append(color_3bit[valid_colors[second_x_index]])
		else:
			color_vector.append(color_3bit[valid_colors[second_x_index]])
			color_vector.append(color_3bit[valid_colors[first_x_index]])		


		# then arrange the rest vertically
		median_sortlist = medians[:, 0].argsort()[::-1]
		count_right = 0
		for m in median_sortlist:
			if medians[m, 0] != 0:
				color_vector.append(color_3bit[valid_colors[m]])
				count_right = count_right + 1
				if count_right > 2:
					break
		ttt = ttt + 1

	# # Case 4:		1-1-1-2 		
	if test_labels_arr[n, :] in [17]:
		print("CASE 4")
		median_sortlist = medians[:, 1].argsort()
		# get leftmost first
		count_left = 0
		for m in median_sortlist:
			if medians[m, 1] != 0:
				color_vector.append(color_3bit[valid_colors[m]])
				medians[m, :] = 0
				count_left = count_left + 1
				if count_left > 2:
					break
			# medians[m, :] = 0

		# then arrange the rest vertically
		median_sortlist = medians[:, 0].argsort()[::-1]
		count_right = 0
		for m in median_sortlist:
			if medians[m, 0] != 0:
				color_vector.append(color_3bit[valid_colors[m]])
				count_right = count_right + 1
				if count_right > 1:
					break
		ttt = ttt + 1

	

	color_vector_flat = [item for sublist in color_vector for item in sublist]
	color_vector_flat = color_vector_flat + [0]*(15-len(color_vector_flat))

	print(test_labels_color[n, :])
	print(color_vector_flat)


	this_file = str(test_list[n])
	this_file = this_file.split('/')[-1]
	with open('predicted_colors.csv', 'a', encoding="utf-8") as f:
		writer = csv.writer(f)
		new_row = [this_file] + color_vector_flat
		writer.writerow(new_row)



	if np.array_equal(np.array(color_vector_flat), np.array(test_labels_color[n, :])):
		ecc = ecc+1


	if ttt > 0:
		print("~~~~ ecc= ", ecc, "~~~~ nott", nott, "~~~~ total= ", ttt, "Test Accuracy= ", 100*ecc/ttt)
		sys.stdout.flush()











