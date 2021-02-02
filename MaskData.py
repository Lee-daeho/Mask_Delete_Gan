from torchvision import datasets, transforms
from torch.utils.data import Dataset

from PIL import Image
import os


class MaskDataset(Dataset):
    def __init__(self, path, label_path, transform):
        self.path = path
        self.label_path = label_path
        self.img_lists = os.listdir(path)
        self.img_lists.sort()
        self.transform = transform

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        image = image_loader(self.path + self.img_lists[idx])
        label = bin_image_loader(self.label_path + self.img_lists[idx])

        return self.transform(image), self.transform(label)


def image_loader(img_name):
    image = Image.open(img_name).convert('RGB')

    return image


def bin_image_loader(img_name):
    image = Image.open(img_name).convert('1')

    return image

class