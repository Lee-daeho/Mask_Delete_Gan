from MaskModule import autoencoder

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

    all_imgs = []
    labels = []

    # for name in img_list:
    #     img = image_loader(IMG_FOLDER + name)
    #     all_imgs.append(img)
    #     b_mask = bin_image_loader(MASK_FOLDER + name)
    #     labels.append(b_mask)



    # all_imgs = np.asarray(all_imgs)
    # labels = np.asarray(labels)
    # print(all_imgs.shape)
    # print(labels.shape)

    train_set = MaskDataset(IMG_FOLDER, MASK_FOLDER, transform)
    print(train_set.__len__())
    print('Image Load End')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False)

    model = autoencoder().to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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

            avg_cost += cost/2

        print('[Epoch:{}] cost = {}'.format(epoch + 1, avg_cost))

        if epoch % 9 == 0:
            print('output shape : ', outputs[0])
            print('save img')
            save_image(outputs[44], RES_IMG_PATH + '/{}.jpg'.format(epoch))
        if epoch % 99 == 0 and not epoch == 0:
            torch.save(model, RES_IMG_PATH + '{}_conv_autoencoder.pkl'.format(epoch))

train()