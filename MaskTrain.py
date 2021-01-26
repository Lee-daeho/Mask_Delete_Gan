from MaskModule import autoencoder
from MaskData import MaskDataset

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import nn

from torchvision.utils import save_image

from PIL import Image

import os

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_EPOCHS = 200


transform = transforms.Compose([
    transforms.ToTensor()
])

def train():
    IMG_FOLDER = './data/imgs/ToDaeHoLee/_masked/'
    MASK_FOLDER = './data/imgs/ToDaeHoLee/_binary/'
    RES_IMG_PATH = './results/'

    if not os.path.exists(RES_IMG_PATH):
        os.makedirs(RES_IMG_PATH)

    img_list = os.listdir(IMG_FOLDER)
    # mask_list = os.listdir('./data/masks')

    train_set = MaskDataset(IMG_FOLDER, MASK_FOLDER, transform)
    print(train_set.__len__())
    print('Image Load End')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False)

    total_batch = len(train_loader)

    model = autoencoder().to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHS):
        avg_cost = 0
        i = 0

        for data in train_loader:
            X = data[0].to(device)
            Y = data[1].to(device)

            optimizer.zero_grad()
            outputs = model(X)

            cost = criterion(outputs, Y)
            cost.backward()

            optimizer.step()

            avg_cost += cost/total_batch

        print('[Epoch:{}] cost = {}'.format(epoch + 1, avg_cost))

        if epoch % 9 == 0:
            print('output shape : ', outputs[0])
            print('save img')
            save_image(outputs[44], RES_IMG_PATH + '/{}.jpg'.format(epoch))
        if epoch % 99 == 0 and not epoch == 0:
            torch.save(model, RES_IMG_PATH + '{}_conv_autoencoder.pkl'.format(epoch))

train()