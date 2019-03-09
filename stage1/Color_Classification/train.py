'''
Script for training Color Classifier on Million Blocks Dataset
inspired from Jacob Fang's Resnet trained on EbayColor Dataset
https://github.com/ASU-Active-Perception-Group/Color_Classification
'''

# 0. IMPORTS
# -------------------------------------
from __future__ import print_function

import os
import sys
import numpy as np 
import csv
import glob 
import time
import datetime
from PIL import Image 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cv2

import torch 
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from lib.net_util import *
from utils.parser import *
from utils.utils import * # utils for arrangement 
from lib.dataset.Ebay_dataset_copy import EbayColor
from lib.dataset.MBD import MillionBlocksDataset
from models.resnet import resnet50
from models.arr_net import ConvNet # network for arrangement

# device stuff --- whatever works!
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparams for ARR
num_epochs = 5
num_classes = 19
batch_size = 4
test_batch_size = 4
learning_rate = 0.001


color_dict = [[255,255,255], [0,0,255], [255,255,255], [0,128,128], [255,255,255], [255,165,0],
		 	  [255,255,255], [128,0,128], [255,0,0], [255,255,255], [255,255,0]]

color_3bit = [[0,0,0], [0,1,1], [0,0,0], [0,1,0], [0,0,0], [1,1,0], 
			  [0,0,0], [1,0,1], [0,0,1], [0,0,0], [1,0,0]]

color_list = []
for i in range(11):
	color_list.append([item/255. for item in color_dict[i]])


def pixel_loss(conv_feat, mask, label, one_hot, model):
	batch_total = 0
	correct = 0
	batch_loss = 0
	# Iterating through batches
	for b in range(conv_feat.shape[0]):
		# Iterating through pixels embeddings
		for i in range(conv_feat.shape[-2]):
			for j in range(conv_feat.shape[-1]):
				# Penalize the pixels of interests
				if mask[b, 0, i, j] != 0:
					pixel_feat = conv_feat[b, :, i, j]
					pixel_feat = pixel_feat.contiguous().view(1, -1)
					prediction = model.fc(pixel_feat)
					batch_loss += opts.criterion[1](torch.squeeze(prediction), one_hot[b])
					# Computing the precision
					_, predicted = torch.max(prediction.data, 1)
					correct += predicted.eq(label[b].data).cpu().sum()
					batch_total += 1
	batch_loss /= batch_total
	return batch_loss, batch_total, correct



def predict_color(conv_feat2, model):
	result = np.zeros((batch_size, 128, 128, 3))
	dilation = np.zeros((batch_size, 128, 128, 3))
	kernel = np.ones((5,5),np.uint8)
	# Iterating through pixels embeddings
	for i in range(conv_feat2.shape[-2]):
		for j in range(conv_feat2.shape[-1]):
			# Penalize the pixels of interests
			pixel_feat = conv_feat2[:, :, i, j]
			pixel_feat = pixel_feat.contiguous().view(batch_size, -1)
			prediction = model.fc(pixel_feat)
			_, predicted = torch.max(prediction.data, 1)
			predicted = predicted.cpu().numpy()
			# print(predicted)
			sys.stdout.flush()

			for b in range(batch_size):
				result[b, i,j, :] = [item/255. for item in color_dict[predicted[b]]]

	for b in range(batch_size):			
		dilation[b, :, :, :] = cv2.dilate(np.squeeze(result[b, :, :, :]) ,kernel, iterations=1)

	return result, dilation



def get_centroids(dilation):
	centroids = np.zeros((batch_size, 11, 2))
	counts = np.zeros((batch_size, 11,1))
	lamda = 1


	for i in range(dilation.shape[0]):
		for j in range(dilation.shape[1]):

			if list(dilation[i, j, :]) in color_list:
				color_idx = color_list.index(list(dilation[i, j, :]))

				centroids[color_idx, 0] = centroids[color_idx, 0] + i
				centroids[color_idx, 1] = centroids[color_idx, 1] + j

				counts[color_idx, 0] = counts[color_idx, 0] + 1

	centroids = np.divide(centroids, counts + lamda)

	return centroids



def centroids2color(centroids):
	centroids = np.nan_to_num(centroids)
	gridified_centroids = np.ceil(centroids/12.8).astype(int) 


	color_vector = [] 

	for b in range(centroids.shape[0]):
		count = 0
		for y in range(1,11):
			for x in range(10, 0, -1):
				varr = np.equal(np.squeeze(gridified_centroids[b, :, :]), [x,y]).all(1)

				if np.any(varr):
					which_color = np.where(varr)[0][0]
					# print(x, y)
					# print(which_color)
					if which_color not in [0,2,4,6,9]:
						color_vector.append(color_3bit[which_color])
						count = count + 1
		color_vector.append([0]*(3*(5-count)))
		if count >5:
			color_vector = color_vector[b*15:(b+1)*15]

		# print(color_vector)

	color_vector_flat = [item for sublist in color_vector for item in sublist]
	# print("len(color_vector_flat)", len(color_vector_flat))
	# color_vector_flat = color_vector_flat + [0]*(15-len(color_vector_flat))

	return np.reshape(color_vector_flat, (centroids.shape[0], -1), 'C') 



def centroids2arrangement(centroids):
	centroids = np.nan_to_num(centroids)
	gridified_centroids = np.ceil(centroids/12.8).astype(int) 

	arr_vector = np.zeros((centroids.shape[0], 5,5))
	for b in range(centroids.shape[0]):
		for y in range(1,11):
			for x in range(10, 0, -1):
				varr = np.equal(np.squeeze(gridified_centroids[b, :, :]), [x,y]).all(1)

				# print(varr)
				if np.any(varr):
					which_color = np.where(varr)[0][0]				

					if which_color not in [0,2,4,6,9]:
						coords = gridified_centroids[b, which_color]
						arr_vector[b, int(coords[0]/2), int(coords[1]/2)] = 1

	arr_vector = np.reshape(arr_vector, (centroids.shape[0], 25))
	# print(arr_vector.shape)

	return arr_vector



def train_net(net, arr_model, opts):

	print('training at epoch {}'.format(opts.epoch+1))

	if opts.use_gpu:
		net.cuda()

	net.train(True)
	train_loss = 0
	total_time = 0
	batch_idx = 0
	optimizer = opts.current_optimizer
	end_time = time.time()
	fig = plt.figure()
	total = 0
	count = 0

	for batch_idx, (images, mask, color_onehot, color_label, im_batch) in enumerate(data_loader):
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# from ARRANGEMENT !!!!!!
		
		images_arr = im_batch['image_arr']
		images_arr = images_arr.to(device)

		labels = im_batch['arrangement']
		labels = labels.reshape(-1, num_classes)
		labels = labels.float().to(device)

		colors = im_batch['color']
		colors = colors.reshape(-1, 15)
		colors = colors.float().to(device)

		outputs = arr_model(images_arr)

		color_label = Variable(color_label).cuda()
		color_onehot = Variable(color_onehot).cuda().float()
		# torch.Size([4, 19])
		# try:
		# 	print(outputs.shape)
		# except:
		# 	print(outputs.size())
		# try:
		# 	print(labels.shape)
		# except:
		# 	print(labels.size())		
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		### ---- EBAY/JACOB ----
		images = Variable(images).cuda()
		colors = Variable(colors).cuda()

		# Feed Forward to Backbone Net
		conv_feat4, conv_feat2 = net(images)

		# get A and C here
		# 4*128*128*3
		result, dilation = predict_color(conv_feat2, net)

		
		# try:
		# 	print(dilation.shape)
		# except:
		# 	print(dilation.size())	

		# 4 * 25
		centroids = get_centroids(dilation)
		# try:
		# 	print(centroids.shape)
		# except:
		# 	print(centroids.size())

		mask_arr = centroids2arrangement(centroids)
		mask_arr = torch.from_numpy(mask_arr).to(device)

		mask_col = centroids2color(centroids)
		# try:
		# 	print(mask_col.shape)
		# except:
		# 	print(mask_col.size())
		mask_col = torch.from_numpy(mask_col).to(device)

		
		# print("arr_of_mask", mask_arr)
		# print("col_of_mask", mask_col)


		# loss_A = opts.criterion[0] (torch.squeeze(outputs), torch.max(labels, 1)[1])
		loss_A = opts.criterion[0] (torch.squeeze(outputs), torch.max(mask_arr, 1)[1])
		loss_C = opts.criterion[0] (torch.squeeze(colors), torch.max(mask_col, 1)[1])

		# Reshape the Binary Mask
		mask = torch.nn.functional.adaptive_avg_pool2d(mask, (conv_feat2.shape[-1], conv_feat2.shape[-2]))

		# Pixel Penalizing
		loss, batch_total, batch_correct = pixel_loss(conv_feat2, mask, color_label, color_onehot, net)

		opts.correct = opts.correct + batch_correct
		opts.total = opts.total + batch_total

		total_loss = loss.data[0] + (opts.alpha * loss_A.data[0]) + (opts.beta * loss_C.data[0])
		train_loss = train_loss + total_loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_idx = batch_idx + 1

		
		if batch_idx % 10 == 0:
			print('---- LOSS: pixel_loss= %.4f , arr_loss= %.4f , col_loss= %.4f' % (loss.data, loss_A.data, loss_C.data))
			sys.stdout.flush()
			writer.add_scalar('MSE Loss', train_loss / (batch_idx + 1), opts.iter_n)
			writer.add_scalar('Precision', opts.correct/opts.total, opts.iter_n)
			opts.iter_n += 10

	train_loss = train_loss / (batch_idx + 1)
	opts.train_epoch_logger.log({'epoch': (opts.epoch+1),
								 'loss': train_loss,
								 'time': total_time})
	opts.train_losses.append(train_loss)

	# Save checkpoint.
	net_states = {
					'state_dict': net.state_dict(),
					'epoch': opts.epoch + 1,
					'loss': opts.train_losses,
					'optimizer': opts.current_optimizer.state_dict() }

	if opts.epoch % opts.checkpoint_epoch == 0:
		save_file_path = os.path.join(opts.checkpoint_path, 'Color_pretrain_regression_{}.pth'.format(opts.epoch))
		torch.save(net_states, save_file_path)

	print('Batch Loss: %.8f, elapsed time: %3.f seconds.' % (train_loss, total_time))



if __name__ == '__main__':
	# parse options
	opts = parse_opts()
	writer = SummaryWriter()

	# gpu stuff
	if opts.gpu_id >= 0:
		torch.cuda.set_device(opts.gpu_id)
		opts.multi_gpu = False

	torch.manual_seed(opts.seed)
	if opts.use_gpu:
		torch.set_default_tensor_type('torch.cuda.FloatTensor')
		torch.cuda.manual_seed(opts.seed)

	# Loading Data
	print("===> Preparing Million Blocks Dataset ===>")
	sys.stdout.flush()
	opts.correct = 0
	opts.total = 0
	size = (512, 512)
	feat_size = (32, 32)
	transform = transforms.Compose([transforms.Resize(size), 
									transforms.ToTensor()])

	

	# --- path/filename
	train_list, test_list, eval_list = get_file_list()

	### EDIT: 2018/10/25
	arr_code_dict ={}
	with open('../../data/arrangement_class.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			key = row[0]
			arr_code_dict[key] = int(row[-1])

	color_code_dict = {}
	with open('../../data/color_code.csv') as f:
		reader = csv.reader(f)
		for row in reader:
			key = row[0]
			rest = row[1:]
			ccode = [float(i) for i in rest]
			color_code_dict[key] = np.array(ccode)

	# NEW
	# ----
	data_set = EbayColor(root_dir='./blocks_data/',
						 feat_size=feat_size,
						 arr_code_dict=arr_code_dict,
						 color_code_dict=color_code_dict,
						 transform=transform)
	data_loader = torch.utils.data.DataLoader(data_set, 
											  batch_size=opts.batch_size, 
											  shuffle=True)


	print("---- DATA LOADED ----")
	sys.stdout.flush()

	if not os.path.exists(opts.result_path):
		os.mkdir(opts.result_path)

	opts.train_epoch_logger = Logger(os.path.join(opts.result_path, 'train' + str(opts.lr) + '_' + str(opts.batch_size) + '.log'),
									 ['epoch', 'time', 'loss'])
	opts.train_batch_logger = Logger(os.path.join(opts.result_path, 'train_batch.log'),
									 ['epoch', 'batch', 'loss'])
	opts.test_epoch_logger = Logger(os.path.join(opts.result_path, 'test.log'),
									['epoch', 'time', 'loss'])

	# Models
	# ---- COLOR RESNET ----
	model = resnet50(False)
	model.fc = torch.nn.Linear(512, 11)
	print('---- COLOR Model Built ----')
	sys.stdout.flush()

	# ---- ARRANGEMENT CONVNET ----
	arr_model = ConvNet(num_classes).to(device)
	# infer arrangment outputs (for TRAIN data)
	PATH = '../Arrangement_Classification/saved_models/model_arr_final.ckpt'
	arr_model.load_state_dict(torch.load(PATH))
	arr_model.eval() 
	print('---- ARRANGEMENT Model Built ----')
	sys.stdout.flush()

	# # EVALUATE ARRANGEMENT NETWORK
	# with torch.no_grad():
	# 	total = 0
	# 	count = 0
	# 	for i, im_batch in enumerate(inference_loader):
	# 		images = im_batch['image']
	# 		images = images.to(device)

	# 		labels = im_batch['arrangement']
	# 		labels = labels.reshape(-1, num_classes)
	# 		labels = labels.float().to(device)

	# 		outputs = arr_model(images)

	# 		outputs.reshape(test_batch_size, num_classes)
	# 		# get class index from one-hot
	# 		_, predicted = torch.max(outputs.data, 1)
	# 		_, classes = torch.max(labels.data, 1)
	# 		total += outputs.size(0)
		   
	# 		if predicted==classes:
	# 			count = count + 1  

	# # print('Train Accuracy of the model on the', total,  
	# # 	  'test images: {} %'.format(100 * count / total))


	# TRAIN RESNET
	if opts.resume:
		state_dict = torch.load(opts.resume)['state_dict']
		new_params = model.state_dict()
		new_params.update(state_dict)
		model.load_state_dict(new_params)

	start_epoch = 0
	opts.criterion = [torch.nn.CrossEntropyLoss(), torch.nn.MSELoss()]

	# Training
	parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in parameters])
	print(params, 'trainable parameters in the network.')
	set_parameters(opts)
	opts.iter_n = 0

	for epoch in range(start_epoch, start_epoch+opts.n_epoch):
		opts.epoch = epoch
		if epoch is 0:
			params = filter(lambda p: p.requires_grad, model.parameters())
			opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

		elif (epoch % opts.lr_adjust_epoch) == 0 and epoch is not 0:
			opts.lr /= 10
			params = filter(lambda p: p.requires_grad, model.parameters())
			opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

		train_net(model, arr_model, opts)

	writer.export_scalars_to_json("./all_scalars.json")
	writer.close()