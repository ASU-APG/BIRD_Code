# IMPORTS
import os
import sys
import numpy as np 
import csv
import glob 
import time
import datetime
from PIL import Image 

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils import *
from data_loader import MillionBlocksDataset 
from model import ConvNet

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 20
num_classes = 19
batch_size = 4
test_batch_size = 1
learning_rate = 0.001

# load data
# ----------------
# --- path/filename
train_list, test_list, eval_list = get_file_list('../../data/all_256_256/', 
                                                 '../../data/test_256_256/', 
                                                 '../../data/eval_256_256/')
# --- images
train_data, test_data, eval_data = load_img_data(train_list, test_list, eval_list)
# --- one-hot labels
train_labels_arr ,test_labels_arr, eval_labels_arr = load_labels_arr(train_list, test_list, eval_list)

# Torch DataLoader
train_dataset = MillionBlocksDataset(labels_arr=train_labels_arr, images = train_data)
test_dataset = MillionBlocksDataset(labels_arr=test_labels_arr, images=test_data)
eval_dataset = MillionBlocksDataset(labels_arr=eval_labels_arr, images=eval_data)
train_loader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
eval_loader = DataLoader(eval_dataset, batch_size=test_batch_size, shuffle=False)

# ConvNet model
model = ConvNet(num_classes).to(device)
model.load_state_dict(torch.load('./saved_models/model_arr_final.ckpt'))

# Test the model 
# eval mode (batchnorm uses moving mean/var instead of mini-batch mean/var)


class_to_arr_vec_dict = {0: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         1: [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         2: [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         3: [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         4: [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         5: [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         6: [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         7: [1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         8: [1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         9: [1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         10: [1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         11: [1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         12: [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         13: [1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         14: [1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                         15: [1,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         16: [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                         17: [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
                         18: [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0]
                         }
for settt in range(1):
    if settt==0:
        d_loader = train_loader
        d_list = train_list
        d_fname = 'predicted_classes_all.csv'
        d_set = 'TRAIN'

    if settt==1:
        d_loader = test_loader
        d_list = test_list
        d_fname = 'predicted_classes_test.csv'
        d_set = 'TEST'

    if settt==2:
        d_loader = eval_loader
        d_list = eval_list
        d_fname = 'predicted_classes_eval.csv'
        d_set = 'EVAL'

    model.eval() 
    with torch.no_grad():
        total = 0
        count = 0
        for i, im_batch in enumerate(d_loader):
            images = im_batch['image']
            images = images.to(device)

            labels = im_batch['arrangement']
            
            labels = labels.reshape(-1, num_classes)
            labels = labels.float().to(device)

            outputs = model(images)
            # print(outputs)

            outputs.reshape(test_batch_size, num_classes)
            # get class index from one-hot
            _, predicted = torch.max(outputs.data, 1)
            _, classes = torch.max(labels.data, 1)
            total = total + outputs.size(0)

            # print("predicted", predicted)
            # print("classes", classes)

            if predicted==classes:
                count = count + 1 

            # predicted = ''.join([str(int(x)) for x in predicted])
            this_file = str(d_list[i]).split('/')[-1]

            
            with open(d_fname, 'a', encoding="utf-8") as f:
                writer = csv.writer(f)
                new_row = [this_file] + class_to_arr_vec_dict[predicted.item()]
                writer.writerow(new_row)
            # print(labels)
            # sys.stdout.flush()
           
     

        print('Test Accuracy of the model on the {} set: {} %'.format(d_set, 100 * count / total))
        sys.stdout.flush()

# with torch.no_grad():
#     total = 0
#     count = 0
#     for i, im_batch in enumerate(eval_loader):
#         images = im_batch['image']
#         images = images.to(device)

#         labels = im_batch['arrangement']
#         # print(labels)
#         labels = labels.reshape(-1, num_classes)
#         labels = labels.float().to(device)

#         outputs = model(images)
#         # print(outputs)

#         outputs.reshape(test_batch_size, num_classes)
#         # get class index from one-hot
#         _, predicted = torch.max(outputs.data, 1)
#         _, classes = torch.max(labels.data, 1)
#         total = total + outputs.size(0)

#         # print("predicted", predicted)
#         # print("classes", /classes)
#         # sys.stdout.flush()
       
#         if predicted==classes:
#             count = count + 1  

#     print('Test Accuracy of the model on the 1453 natural images: {} %'.format(100 * count / total))
#     sys.stdout.flush()