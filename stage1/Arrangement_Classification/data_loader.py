
from torch.utils.data import Dataset, DataLoader

class MillionBlocksDataset(Dataset):
    """Million Blocks dataset."""

    def __init__(self, labels_arr, images, transform=None):

        self.arr_code = labels_arr 
        self.images = images # train_list 
        self.transform = transform

    def __len__(self):
        return len(self.arr_code)

    def __getitem__(self, idx):

        image = self.images[idx]
        arrangement = self.arr_code[idx] 
        sample = {'image': image, 'arrangement': arrangement}

        if self.transform:
            sample = self.transform(sample)

        return sample 