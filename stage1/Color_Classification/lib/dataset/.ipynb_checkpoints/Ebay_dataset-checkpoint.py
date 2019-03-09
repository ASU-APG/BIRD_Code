import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# Preparing the color image into dataset class
# Returning:
# (Image, Mask, Color)
# Image and mask would be in the same size, and color will be converted to a one-hot format

class EbayColor(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, feat_size, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        imag_list = []
        mask_list = []
        imag_dir = os.path.join(root_dir, 'test_images.txt')
        mask_dir = os.path.join(root_dir, 'mask_images.txt')

        # Labels
        labels = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink',
                  'purple', 'red', 'white', 'yellow']
        # Read Image directories
        list_file = open(imag_dir, 'r')
        for i in list_file.readlines():
            imag_list.append(os.path.join(root_dir, i[0:-1].replace('./', '')))

        # Read Mask directories
        list_file = open(mask_dir, 'r')
        for i in list_file.readlines():
            mask_list.append(os.path.join(root_dir, i[0:-1].replace('./', '')))

        self.img_files = imag_list
        self.mask_files = mask_list
        self.length = len(self.img_files)
        self.root_dir = root_dir
        self.transform = transform
        self.feat_size = feat_size
        self.labels = labels

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        imag_direc = self.img_files[index]
        mask_direc = self.mask_files[index]

        image = Image.open(imag_direc)
        image = image.convert('RGB')
        mask = Image.open(mask_direc)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        # Parse out the color label
        color = imag_direc.split('/')[-2]

        # Building up the one-hot label
        label_index = self.labels.index(color)
        one_hot = np.zeros(len(self.labels))
        one_hot[label_index] = 1
        return image, mask, one_hot, label_index