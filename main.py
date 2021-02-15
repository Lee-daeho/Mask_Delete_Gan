from MaskModule import autoencoder
from MaskData import MaskDataset, EditDataset
from EditModule import Discriminator, Generator

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn

import pytorch_ssim

from torchvision.utils import save_image

from PIL import Image
from tqdm import tqdm

import os

import numpy as np

from argparse import ArgumentParser

from util import *

from argparse import ArgumentParser

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.ToTensor()
])

def train(args):
    IMG_FOLDER = './data/imgs/train/_masked/'
    MASK_FOLDER = './data/imgs/train/_binary/'
    ORG_FOLDER = './data/imgs/train/_origin/'
    RES_IMG_PATH = args.dir_result
    CKP_DIR = args.dir_checkpoint

    if not os.path.exists(RES_IMG_PATH):
        os.makedirs(RES_IMG_PATH)

    if not os.path.exists(CKP_DIR):
        os.makedirs(CKP_DIR)

    img_list = os.listdir(IMG_FOLDER)

    train_set = EditDataset(args.dir_data + '/_masked/', args.dir_data + '/_origin/', args.dir_data + '/_binary/',
                                transform)

    learning_rate = args.lr
    NUM_EPOCHS = args.epoch
    BATCH_SIZE = args.BATCH_SIZE

    print(train_set.__len__())
    print('Image Load End')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size= BATCH_SIZE, shuffle=False)

    total_batch = len(train_loader)

    netG = Generator().to(device)
    netD_whole = Discriminator(in_ch=2*3, out_ch=1, nker=64, norm='bnorm').to(device)
    netD_mask = Discriminator(in_ch=2*3, out_ch=1, nker=64, norm='bnorm').to(device)

    criterion_L1 = nn.L1Loss().to(device)
    criterion_SSIM = pytorch_ssim.SSIM().to(device)
    criterion_BCE = nn.BCEWithLogitsLoss().to(device)

    optimG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D_whole = torch.optim.Adam(netD_whole.parameters(), lr=learning_rate)
    optimizer_D_mask = torch.optim.Adam(netD_mask.parameters(), lr=learning_rate)

    ######################### load model #################################

    for epoch in range(NUM_EPOCHS):
        avg_cost = 0
        i = 0
        netG.train()
        if epoch < 0.4 * NUM_EPOCHS:
            netD_whole.train()
        else:
            netD_mask.train()

        loss_D_whole_real_train = []
        loss_D_whole_fake_train = []
        loss_D_mask_real_train = []
        loss_D_mask_fake_train = []

        for batch, data in enumerate(train_loader):
            X = data[0].to(device)
            Y = data[1].to(device)
            ############################## get ground truth data(I_gt) through dataloader. ##############################

            if epoch < 0.4 * NUM_EPOCHS:
                set_requires_grad(netD_whole, True)
                optimizer_D_whole.zero_grad()
            else:
                set_requires_grad(netD_mask, True)
                optimizer_D_mask.zero_grad()

            # backward D_Whole_region
            # output: generator로 생성된 이미지(I_edit)

            output = netG(X)

            # real_whole = torch.cat([X[:,0:2,:,:], Y], dim=1)
            # fake_whole = torch.cat([X[:,0:2,:,:], output], dim=1)
            # real_mask = torch.cat([X[:,0:2,:,:]*X[:,3,:,:], Y*X[:,3,:,:]], dim=1)
            # fake_mask = torch.cat([X[:,0:2,:,:]*X[:,3,:,:], output*X[:,3,:,:]], dim=1)

            mask_region_1D = torch.unsqueeze(X[:,3,:,:],1)
            mask_region = torch.cat([mask_region_1D,mask_region_1D,mask_region_1D], dim=1)

            real_whole = torch.cat([X[:,0:3,:,:], Y], dim=1)
            fake_whole = torch.cat([X[:,0:3,:,:], output], dim=1)
            real_mask = torch.cat([X[:,0:3,:,:].mul(mask_region), Y.mul(mask_region)], dim=1)
            fake_mask = torch.cat([X[:,0:3,:,:].mul(mask_region), output.mul(mask_region)], dim=1)

            pred_real_whole = netD_whole(real_whole)
            pred_fake_whole = netD_whole(fake_whole.detach())
            pred_real_mask = netD_mask(real_mask)
            pred_fake_mask = netD_mask(fake_mask.detach())

            loss_D_whole_real = criterion_BCE(pred_real_whole, torch.ones_like(pred_real_whole))
            loss_D_whole_fake = criterion_BCE(pred_fake_whole, torch.zeros_like(pred_fake_whole))
            loss_whole_D = 0.5 * (loss_D_whole_real + loss_D_whole_fake)

            loss_D_mask_real = criterion_BCE(pred_real_mask, torch.ones_like(pred_real_mask))
            loss_D_mask_fake = criterion_BCE(pred_fake_mask, torch.zeros_like(pred_fake_mask))
            loss_mask_D = 0.5 * (loss_D_mask_real + loss_D_mask_fake)

            if epoch < 0.4 * NUM_EPOCHS:
                loss_whole_D.backward(retain_graph=True)
                optimizer_D_whole.step()
            else:
                loss_mask_D.backward(retain_graph=True)
                optimizer_D_mask.step()

            if epoch < 0.4 * NUM_EPOCHS:
                set_requires_grad(netD_whole, False)
                optimizer_D_whole.zero_grad()
            else:
                set_requires_grad(netD_mask, False)
                optimizer_D_mask.zero_grad()

            L_rc = criterion_L1(output, Y) + (1 - criterion_SSIM(output, Y))

            loss_comp = L_rc*100 + 0.6 * loss_whole_D + 1.4 * loss_mask_D

            loss_comp.backward()
            optimG.step()

            loss_D_whole_real_train += [loss_D_whole_real.item()]
            loss_D_whole_fake_train += [loss_D_whole_fake.item()]
            loss_D_mask_real_train += [loss_D_mask_real.item()]
            loss_D_mask_fake_train += [loss_D_mask_fake.item()]

            print("TRAIN: EPOCH %04d | BATCH %04d / %04d | "
                  "L_D_whole_r: %.4f | L_D_whole_f: %.4f | "
                  "L_D_mask_r: %.4f | L_D_mask_f: %.4f" %
                  (epoch, batch, total_batch,
                   np.mean(loss_D_whole_real_train), np.mean(loss_D_whole_fake_train),
                   np.mean(loss_D_mask_real_train), np.mean(loss_D_mask_fake_train)))

            # avg_cost += cost/total_batch

        # print('[Epoch:{}] cost = {}'.format(epoch + 1, avg_cost))

        if epoch % 9 == 0:
            print('output shape : ', output[0])
            print('save img')
            for i in tqdm(range(X.shape[0]), desc='saving'):
                save_image(output[i], RES_IMG_PATH + '{}epoch_{}.jpg'.format(epoch, i))
        if epoch % 99 == 0 and not epoch == 0:
            torch.save(netG, CKP_DIR + '{}_netG.pkl'.format(epoch))

if __name__ == "__main__":

    parser = ArgumentParser(description='mask module')

    parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')

    parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
    parser.add_argument('--dir_data', default='./data/imgs/train/')

    parser.add_argument('--epoch', default=50, dest='epoch', type=int)
    parser.add_argument('--learning_rate', default=1e2, dest='lr', type=float)
    parser.add_argument('--batch_size', default=2, dest='BATCH_SIZE', type=int)

    parser.add_argument('--dir_result', default='./final_results', dest='dir_result')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)

    elif args.mode == 'test':
        print('not yet')
