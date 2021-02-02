from MaskModule import autoencoder
from MaskData import MaskDataset, EditDataset

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn

from torchvision.utils import save_image

import EditModule
import pytorch_ssim

from PIL import Image
from tqdm import tqdm

import os

import numpy as np

from argparse import ArgumentParser

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = ArgumentParser(description = 'mask module')

parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')

parser.add_argument('--dir_checkpoint', default='./checkpoints', dest = 'dir_checkpoint')
parser.add_argument('--dir_data', default='./data/imgs/train/')

parser.add_argument('--epoch', default=100000, dest='epoch')
parser.add_argument('--learning_rate', default=1e2, dest='lr')
parser.add_argument('--epoch', default=100000, dest='epoch')
parser.add_argument('--epoch', default=100000, dest='epoch')

parser.add_argument('--dir_result', default='./final_results', dest='dir_result')

args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor()
])

def main():

    if args.mode == 'train':
        train_dataset = EditDataset(args.dir_data + '/_masked/', args.dir_data + '/_origin', args.dir_data + '/_binary/', transform)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

        learning_rate = args.lr
        epoch = args.epoch

        netG = EditModule.Generator().to(device)
        whoel_netD = EditModule.Discriminator(in_ch=6, out_ch=1, nker=64).to(device)
        mask_netD = EditModule.Discriminator(in_ch=6, out_ch=1, nker=64).to(device)

        paramsG = netG.parameters()
        whole_paramsD = whoel_netD.parameters()
        mask_paramsD = mask_netD.parameters()

        optimG = torch.optim.Adam(paramsG, lr=learning_rate, betas=(0.5, 0.999))
        mask_optimD = torch.optim.Adam(mask_paramsD, lr=learning_rate, betas=(0.5, 0.999))
        whole_optimD = torch.optim.Adam(whole_paramsD, lr=learning_rate, betas=(0.5, 0.999))

        L_l1 = nn.L1Loss().to(device)
        L_ssim = pytorch_ssim.SSIM().to(device)

        for itr in epoch:
            netG.train()
            whoel_netD.train()
            mask_netD.train()

            for i, data in enumerate(train_loader):

                X = data[0].to(device)
                Y = data[1].to(device)

                output = netG(X)

                fake = torch.cat([X, output], dim=1)
                real = torch.cat([X, Y], dim=1)

                mask_fake = torch.cat([])
                whole_paramsD.requires_grad = True
                mask_paramsD.requires_grad = True

                whole_optimD.zero_grad()
                mask_optimD.zero_grad()






    elif args.mode == 'test':
        pass


if __name__ == '__main__':

    if not os.path.exists(args.dir_result):
        os.mkdir(args.dir_result)

    if not os.path.exists(args.dir_checkpoint):
        os.mkdir(args.dir_checkpoint)

    main()