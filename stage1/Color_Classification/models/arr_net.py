#
import torch 
import torch.nn as nn
import numpy as np 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvNet(nn.Module):
    def __init__(self, num_classes=19):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=5, stride=2),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 1024), 
            nn.ReLU())
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):

        # x = np.transpose(x, (0,3,1,2))
        x = x.to(device)
        # print("x.size()", x.size())

        conv1 = self.layer1(x)
        # print("conv1.size()", conv1.size())

        conv2 = self.layer2(conv1)
        # print("conv2.size()", conv2.size())

        conv3 = self.layer3(conv2)
        # print("conv3.size()", conv3.size())

        conv4 = self.layer4(conv3)
        # print("conv4.size()", conv4.size())

        conv5 = self.layer5(conv4)
        # print("conv5.size()", conv5.size())

        conv6 = self.layer6(conv5)
        # print("conv6.size()", conv6.size())

        lin1 = self.fc1(conv6.reshape(conv6.size(0), -1))
        out = self.fc2(lin1)

        return out
