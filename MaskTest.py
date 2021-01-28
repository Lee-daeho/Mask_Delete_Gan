from MaskModule import autoencoder

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import nn

from MaskTrain import MaskDataset

from torchvision.utils import save_image

from PIL import Image

import os
from tqdm import tqdm
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def image_loader(img_name):
    image = Image.open(img_name).convert('RGB')

    return image


def bin_image_loader(img_name):
    image = Image.open(img_name).convert('1')

    return image


transform = transforms.Compose([
    transforms.ToTensor()
])

def test():
    IMG_FOLDER = './data/imgs/test/_masked/'
    MASK_FOLDER = './data/imgs/test/_binary/'
    RES_IMG_PATH = './test_results/'
    MODEL_PATH = './results/'

    if not os.path.exists(RES_IMG_PATH):
        os.makedirs(RES_IMG_PATH)

    test_set = MaskDataset(IMG_FOLDER, MASK_FOLDER, transform)
    print(test_set.__len__())
    print('Image Load End')

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    model = autoencoder().to(device)

    model = torch.load(MODEL_PATH + '40_conv_autoencoder.pkl')

    data_lists = os.listdir(IMG_FOLDER)
    data_lists.sort()

    for num, data in enumerate(test_loader):
        X = data[0].to(device)
        Y = data[1].to(device)

        outputs = model(X)

        for i in tqdm(range(X.shape[0]), desc='saving'):
            save_image(outputs[i], RES_IMG_PATH + '{}.jpg'.format(num*128 + i))

test()