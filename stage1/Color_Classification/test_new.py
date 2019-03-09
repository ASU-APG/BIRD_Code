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
from scipy import ndimage


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
test_list = glob.glob('../../data/test_256_256/*.jpg')
test_flist = []
for item in test_list:
	test_flist.append(item.split('/')[-1])

test_labels_color = np.zeros((len(test_flist), 15))
num = 0
color_code_dict = {}
with open('../../data/color_code.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		key = row[0]
		rest = row[1:]
		ccode = [float(i) for i in rest]
		color_code_dict[key] = np.array(ccode)

for item in test_flist:
	test_labels_color[num, :] = color_code_dict[item]
	num = num+1

size = (1024, 1024)
transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])

ccar = 0 # correct colors after removal of insertion
ccio = 0 # correct colors in order
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
	# print(colormap)
	plt.imsave('./output/' + test_flist[n][:-4] + '.png', result)

	centroids = np.zeros((11, 2))
	medians = np.zeros((11, 2))
	counts = np.zeros((11,1))

	locs_x = {}
	locs_y = {}
	medians_x = {}
	medians_y = {}


	for key in range(0, 11):
		locs_x[key] = []
		locs_y[key] = []


	masks = np.zeros((128, 128, 11))
	# dilate
	kernel = np.ones((5, 5),np.uint8)

	

	for color_idx in range(11):
		masks[:, :, color_idx] = (colormap==color_idx)*1

		this_mask = np.array(masks[:, :, color_idx], dtype = np.uint8)

		# Find the largest contour and extract it
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

		# if color_idx not in [0,2,4,6,9]:
		# 	plt.imsave('./eroded/' + str(color_idx)+ '_'+ test_flist[n], finalImage) 

		masks[:, :, color_idx] = finalImage

		for i in range(masks.shape[0]):
			for j in range(masks.shape[1]):
				if masks[i, j, color_idx] == 1:

					# centroids[color_idx, 0] = centroids[color_idx, 0] + i
					# centroids[color_idx, 1] = centroids[color_idx, 1] + j
					locs_x[color_idx] = locs_x[color_idx] + [i]
					locs_y[color_idx] = locs_y[color_idx] + [j]

					counts[color_idx, 0] = counts[color_idx, 0] + 1

	

	# centroids = np.divide(centroids, counts)
	for key in range(0, 11):
		if locs_x[key]:
			medians[key, 0] = statistics.median(locs_x[key])
		if locs_y[key]:
			medians[key, 1] = statistics.median(locs_y[key])

	gridified_medians = gridify(medians)
	gridified_centroids = gridified_medians
	# print("medians ", medians)
	# # print("counts", counts )
	# print("gridified_centroids", gridified_centroids)

	color_vector = [] 

	for y in range(1,11):
		for x in range(10, 0, -1):	
			if np.any(np.equal(gridified_centroids, [x,y]).all(1)):
				which_color = np.where(np.equal(gridified_centroids, [x,y]).all(1))[0][0]
				# print(x, y)
				# print(which_color)
				if which_color not in [0,2,4,6,9]:
					color_vector.append(color_3bit[which_color])

	color_vector_flat = [item for sublist in color_vector for item in sublist]
	color_vector_flat = color_vector_flat + [0]*(15-len(color_vector_flat))
	# a
	color_vector_3_5 = np.reshape(color_vector_flat, (-1, 3))
	# b
	test_labels_color_3_5 = np.reshape(test_labels_color[n, :], (-1, 3))
	ttb = ttb + test_labels_color_3_5.shape[0]

	print(test_labels_color[n, :])
	print(color_vector_flat)
	
	if np.array_equal(np.array(color_vector_flat), np.array(test_labels_color[n, :])):
		ecc = ecc+1

	index_of_gt = 0
	for cc in range(len(color_vector_3_5)):
		if color_vector_3_5[cc].tolist() in test_labels_color_3_5.tolist():
			ccc = ccc + 1
			# if np.array_equal(np.array(color_vector_3_5[cc].tolist()), np.array(test_labels_color_3_5[cc].tolist())):
			# 	ccio = ccio + 1
			if np.array_equal(np.array(color_vector_3_5[cc].tolist()), np.array(test_labels_color_3_5[index_of_gt].tolist())):
				ccar = ccar + 1
				index_of_gt = index_of_gt + 1

	ttt = ttt + 1
	print(n, "\t---", ecc/ttt, "\t---", ccc/(5 *ttt), "\t---- ", ccio, "\t---- ", ccar/ttb)
	sys.stdout.flush()
print("ecc: ", ecc, "ccc: ", ccc, "ccio: ", ccio, "ccar: ", ccar )










