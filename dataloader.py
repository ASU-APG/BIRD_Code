import numpy as np
from torch.utils.data.dataset import Dataset
#from torchvision import transforms
import csv
import torch

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
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
        #self.transforms = transform

    def __getitem__(self, index):
        single_label = self.labels[index]
        inp_tensor = self.data[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28]) 
        #img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28,28).astype('uint8')
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        #img_as_img = Image.fromarray(img_as_np)
        #img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        #if self.transforms is not None:
        #    img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return (inp_tensor, single_label)

    def __len__(self):
        return self.data.shape[0]
        
if __name__ == "__main__":
    # Define transforms
    #transformations = transforms.Compose([transforms.ToTensor()])
    # Define custom dataset
    custom_csv = CustomDatasetFromCSV('sample.csv')
    # Define data loader
    dataset_loader = torch.utils.data.DataLoader(dataset=custom_csv,batch_size=100,shuffle=True)
    
    for inputs, labels in dataset_loader:
        print(inputs[0])
        print(labels[0])
        
