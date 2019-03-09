# IMPORTS
import os
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
train_list, test_list, eval_list = get_file_list()
# --- images
train_data, test_data, eval_data = load_img_data(train_list, test_list, eval_list)
# --- one-hot labels
train_labels_arr ,test_labels_arr, eval_labels_arr = load_labels_arr(train_list, test_list, eval_list)

# Torch DataLoader
train_dataset = MillionBlocksDataset(labels_arr=train_labels_arr, images = train_data)
test_dataset = MillionBlocksDataset(labels_arr=test_labels_arr, images=test_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

# ConvNet model
model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, im_batch in enumerate(train_loader):
        images = im_batch['image']
        images = images.to(device)

        labels = im_batch['arrangement']
        labels = labels.reshape(-1, num_classes)
        labels = labels.long().to(device)
     
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model 
# eval mode (batchnorm uses moving mean/var instead of mini-batch mean/var)
model.eval() 
with torch.no_grad():
    total = 0
    count = 0
    for i, im_batch in enumerate(test_loader):
        images = im_batch['image']
        images = images.to(device)

        labels = im_batch['arrangement']
        labels = labels.reshape(-1, num_classes)
        labels = labels.float().to(device)

        outputs = model(images)

        outputs.reshape(test_batch_size, num_classes)
        # get class index from one-hot
        _, predicted = torch.max(outputs.data, 1)
        _, classes = torch.max(labels.data, 1)
        total += outputs.size(0)
       
        if predicted==classes:
            count = count + 1  

    print('Test Accuracy of the model on the 1453 test images: {} %'.format(100 * count / total))

# Save the model checkpoint
# save model
if not os.path.exists('./saved_models/'):
  os.makedirs('./saved_models/')

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

torch.save(model.state_dict(), "./saved_models/model_arr_" + st + ".ckpt")