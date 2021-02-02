import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import nn

from PIL import Image

import os

import numpy as np


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = self.doubleConv(4,16)      #downsample     #ADD SE block
        self.SE1 = self.SENetwork(16, 16)
        self.conv2 = self.doubleConv(16,32)         #ADD SE block
        self.SE2 = self.SENetwork(32, 16)
        self.conv3 = self.doubleConv(32,64)         #ADD SE block
        self.SE3 = self.SENetwork(64, 16)
        self.conv4 = self.doubleConv(64,128)

        self.middle_conv = self.doubleConv(128,256) #middle

        #ADD ATROUS CONVOLUTION BLOCK --> RATE : 2,4,8,16

        self.deconv1 = self.deconv(256,128) #upsample
        self.deconv2 = self.deconv(128,64)
        self.deconv3 = self.deconv(64,32)
        self.deconv4 = self.deconv(32,16)

        self.out = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )       #last layer

        self.upconv1 = self.doubleConv(256,128)     #convolution during upsample
        self.upconv2 = self.doubleConv(128, 64)
        self.upconv3 = self.doubleConv(64, 32)
        self.upconv4 = self.doubleConv(32, 16)

        self.pool = nn.MaxPool2d(2, stride=2)

    def conv(self, in_channel, out_channel):
        convolution = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,stride=1, padding=1),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU()
        )

        return convolution

    def deconv(self, in_channel, out_channel):
        deconvolution = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU()
        )

        return deconvolution

    def doubleConv(self, in_channel, out_channel):
        dConv = nn.Sequential(
            self.conv(in_channel, out_channel),
            self.conv(out_channel, out_channel)
        )

        return dConv

    def SENetwork(self, in_channel, ratio):

        SENet = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Linear(in_channel, int(in_channel/ratio)),
            nn.ReLU(),
            nn.Linear(int(in_channel/ratio), in_channel),
            nn.Sigmoid()
        )

        return SENet


class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=0.0):
        super(CBR2d, self).__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]

        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        if not relu is None and relu >= 0.0:
            layers += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)




class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, nker=64, norm="bnorm"):
        super(Discriminator, self).__init__()

        self.enc1 = CBR2d(1 * in_ch, 1 * nker, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=0.2, bias=False)

        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc4 = CBR2d(4 * nker, 8 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc5 = CBR2d(8 * nker, out_ch, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=None, bias=False)

        def forward(self, x):
            x = self.enc1(x)
            x = self.enc2(x)
            x = self.enc3(x)
            x = self.enc4(x)
            x = self.enc5(x)

            x = torch.sigmoid(x)

            return x
