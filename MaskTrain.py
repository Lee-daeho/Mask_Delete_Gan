from MaskModule import autoencoder
from MaskData import MaskDataset

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import nn

from torchvision.utils import save_image

from PIL import Image
from tqdm import tqdm

import os

import numpy as np

from argparse import ArgumentParser

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
            for i in tqdm(range(X.shape[0]), desc='saving'):
                save_image(outputs[i], RES_IMG_PATH + '{}epoch_{}.jpg'.format(epoch, i))
        if epoch % 99 == 0 and not epoch == 0:
            torch.save(model, RES_IMG_PATH + '{}_conv_autoencoder.pkl'.format(epoch))

def test():
    IMG_FOLDER = './data/imgs/test/_masked/'
    MASK_FOLDER = './data/imgs/test/_binary/'
    RES_IMG_PATH = './test_results_EPOCH198/'
    MODEL_PATH = './results/'

    if not os.path.exists(RES_IMG_PATH):
        os.makedirs(RES_IMG_PATH)

    test_set = MaskDataset(IMG_FOLDER, MASK_FOLDER, transform)
    print(test_set.__len__())
    print('Image Load End')

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

    model = autoencoder().to(device)

    model = torch.load(MODEL_PATH + '198_conv_autoencoder.pkl')

    data_lists = os.listdir(IMG_FOLDER)
    data_lists.sort()

    for num, data in enumerate(test_loader):
        X = data[0].to(device)
        Y = data[1].to(device)

        outputs = model(X)

        for i in tqdm(range(X.shape[0]), desc='saving'):
            save_image(outputs[i], RES_IMG_PATH + data_lists[100*num + i] + '.jpg')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tt',dest='tt', type=str, help = 'type train or test')
    
    args = parser.parse_args()
    
    if args.tt == 'train':
        train()
    elif args.tt == 'test':
        test()
    else:
        print('arg')