import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

  
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)

        x = x.view(x.size(0), 16, 8)
        x = self.softmax(x)
        x = torch.clamp(x, 0.001, torch.max(x).item())
        x = x.view(x.size(0), -1)


        return x

  

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{:02d}.pth'.format(epoch))


class RN(BasicModel):
    def __init__(self):
        super(RN, self).__init__()
        
        self.conv = ConvInputModel()
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24+2)*2+(26*25), 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(64, 2)
        self.coord_oj = torch.FloatTensor(64, 2)
        
            
        self.coord_oi = Variable(self.coord_oi).to(device)
        self.coord_oj = Variable(self.coord_oj).to(device)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(64, 25, 2)

        self.coord_tensor = Variable(self.coord_tensor).to(device)
        np_coord_tensor = np.zeros((64, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()
        



    def forward(self, img, img2):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        x2 = self.conv(img2)

        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]

        # print(d)
        # sys.stdout.flush()





        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        x2_flat = x2.view(mb,n_channels,d*d).permute(0,2,1)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        x2_flat = torch.cat([x2_flat, self.coord_tensor],2)
        
        # add question everywhere
        qst = x2_flat.view(mb, -1) # 64 x 650
        qst = torch.unsqueeze(qst, 1) # 64 x 1 x 650
        qst = qst.repeat(1,25,1)        # 64 x 25 x 650
        qst = torch.unsqueeze(qst, 2)   # 64 x 25 x 1 x 650
        # print("qst.shape", qst.shape)
        # sys.stdout.flush()
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat,1) # (64x1x25x24)
        x_i = x_i.repeat(1,25,1,1) # (64x25x25x24)
        # print("x_i.shape", x_i.shape)
        # sys.stdout.flush()
        x_j = torch.unsqueeze(x_flat,2) # (64x25x1x26)
        # print("x_j.shape", x_j.shape)
        # sys.stdout.flush()
        x_j = torch.cat([x_j,qst],3)
        # print("x_j.shape", x_j.shape)
        # sys.stdout.flush()
        x_j = x_j.repeat(1,1,25,1) # (64x25x25x26+11)
        # print("x_j.shape", x_j.shape)
        # sys.stdout.flush()
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+11)

        # print(x_full.shape)
        # sys.stdout.flush()
        
        # reshape for passing through network
        x_ = x_full.view(mb*d*d*d*d,702)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb,d*d*d*d,256)
        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)


# class CNN_MLP(BasicModel):
#     def __init__(self, args):
#         super(CNN_MLP, self).__init__(args, 'CNNMLP')

#         self.conv  = ConvInputModel()
#         self.fc1   = nn.Linear(5*5*24 + 11, 256)  # question concatenated to all
#         self.fcout = FCOutputModel()

#         self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
#         #print([ a for a in self.parameters() ] )
  
#     def forward(self, img, qst):
#         x = self.conv(img) ## x = (64 x 24 x 5 x 5)

#         """fully connected layers"""
#         x = x.view(x.size(0), -1)
        
#         x_ = torch.cat((x, qst), 1)  # Concat question
        
#         x_ = self.fc1(x_)
#         x_ = F.relu(x_)
        
#         return self.fcout(x_)

