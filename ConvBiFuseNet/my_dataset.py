from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataSet(Dataset):
    """Custom Dataset"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)  # Return the number of images

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        # Convert label to a tensor, ensuring shape is (1,)
        label = torch.tensor(label, dtype=torch.long)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)

        images = torch.stack(images, dim=0)  # Stack images into a tensor
        labels = torch.stack(labels, dim=0)
        return images, labels
