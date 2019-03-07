import torch                                                                    
from torch.autograd import Variable                                             
import torch.nn as nn                                                           
import torch.nn.functional as F                                                 
import torch.optim as optim                                                     
import csv                                                                      
import numpy as np                                                              
from torch.utils.data import DataLoader, Dataset                                
                                                                                
class DatasetLd(Dataset):                                                       
                                                                                
    def __init__(self, file_path):                                               
        self.csv_file = open(file_path,'r')                                     
        self.csv_reader = csv.reader(self.csv_file)                             
        self.data = list(self.csv_reader)                                       
        self.file_path = file_path                                              
                                                                                
    def __len__(self):                                                          
        f = open(self.file_path,'r')                                            
        return sum(1 for i in f) 

    def __getitem__(self, index):                                                
        row = self.data[index]                                                  
        self.inc, self.out = eval(row[0])+eval(row[2]), eval(row[5])            
        self.inc = np.asarray(self.inc)                                         
        self.out = np.asarray(self.out)                                         
        return self.inc, self.out  

def create_nn(batch_size=200, learning_rate=0.01, epochs=50,log_interval=20):   
                                                                                
    train_dataset = DatasetLd('./sampledplans_shuffled_train10k.csv')            
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                                                                                
    test_dataset = DatasetLd('./sampledplans_shuffled_test.csv')                
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 
                                                                                
    class Net(nn.Module):                                                       
        def __init__(self):                                                     
            super(Net, self).__init__()                                         
            self.fc1 = nn.Linear(80, 100)                                       
            self.fc2 = nn.Linear(100, 200)                                      
            self.fc3 = nn.Linear(200, 500)                                      
            self.fc4 = nn.Linear(500, 48) 

        def forward(self, x):                                                   
            x = self.fc1(x)                                                     
            x = self.fc2(x)                                                     
            x = self.fc3(x)                                                     
            x = self.fc4(x)                                                     
            return x 

    net = Net()                                                                 
    print(net)                                                                  
                                                                                
    # create a stochastic gradient descent optimizer                            
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)     
    # create a loss function                                                    
    criterion = nn.CrossEntropyLoss()

    # run the main training loop                                                
    for epoch in range(epochs):                                                 
        for batch_idx, (data, target) in enumerate(train_loader):               
            #print(type(data))                                                  
            #print(type(target))                                                
            data, target = Variable(data.float()), Variable(target.float())     
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)   
            # data = data.view(-1, 28*28)                                       
            #print(data)                                                        
            #print(data.shape)                                                  
            #print(target)                                                      
            #print(target.shape)                                                
            optimizer.zero_grad()                                               
            net_out = net(data)                                                 
            #print(type(net_out))                                               
            #print(net_out.shape)                                               
            #print(net_out[0])                                                  
            loss = criterion(net_out, target)                                   
            loss.backward()                                                     
            optimizer.step()
            if batch_idx % log_interval == 0:                                   
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( 
                    epoch, batch_idx * len(data), len(train_loader.dataset),    
                           100. * batch_idx / len(train_loader), loss.data[0]))

    test_loss = 0                                                               
    correct = 0                                                                 
    for data1, target1 in test_loader:                                          
        #print(data1[0])                                                        
        #print(data1.shape)                                                     
        #print(target1.shape)                                                   
        data1, target1 = Variable(data1.float()), Variable(target1.float())     
        net_out1 = net(data1)                                                   
        # sum up batch loss                                                     
        test_loss += criterion(net_out1, target1).data[0]                       
    test_loss /= len(test_loader.dataset)                                       
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss)) 

create_nn()     

