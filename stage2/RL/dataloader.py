import numpy as np                                                              
from torch.utils.data.dataset import Dataset                                    
#from torchvision import transforms                                             
import csv                                                                      
import torch                                                                    
                                                                                
class CustomDatasetFromCSV(Dataset):                                            
    def __init__(self,csv_path):                                                                                    
        X = []                                                                  
        y = []                                                                  
        csv_file = open(csv_path,'r')                                           
        csv_reader = csv.reader(csv_file)                                       
        for row in csv_reader:                                                  
            inc, out = eval(row[0])+eval(row[2]), eval(row[5])                  
        X.append(inc)                                                           
        y.append(out)                                                           
        csv_file.close()                                                        
        self.data = np.asarray(X)                                               
        self.labels = np.asarray(y)                                             
                                                                                
    def __getitem__(self, index):                                               
        single_label = self.labels[index]                                       
        inp_tensor = self.data[index]                                                                        
        return (inp_tensor, single_label)                                       
                                                                                
    def __len__(self):                                                          
        return self.data.shape[0]                                               
                                                                                
if __name__ == "__main__":                                                      
    custom_csv = CustomDatasetFromCSV('sample.csv')                             
    dataset_loader = torch.utils.data.DataLoader(dataset=custom_csv,batch_size=100,shuffle=True)                                                                          
    for inputs, labels in dataset_loader:                                       
        print(inputs[0])                                                        
        print(labels[0])