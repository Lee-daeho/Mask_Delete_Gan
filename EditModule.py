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
        self.at_conv1 = self.AtrousConv(256,256,2)
        self.at_conv2 = self.AtrousConv(256, 256, 4)
        self.at_conv3 = self.AtrousConv(256, 256, 8)
        self.at_conv4 = self.AtrousConv(256, 256, 16)

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

    def AtrousConv(self, in_channel, out_channel, dilate):

        AConv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=dilate, dilation=dilate),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU()
        )

        return AConv

    def forward(self, x):
        down1 = self.conv1(x)       #16, 256, 256
        pool1 = self.pool(down1)    #16, 128, 128
        down2 = self.conv2(pool1)   #32, 128, 128
        pool2 = self.pool(down2)    #32, 64, 64
        down3 = self.conv3(pool2)   #64, 64, 64
        pool3 = self.pool(down3)    #64, 32, 32
        down4 = self.conv4(pool3)   #128, 32, 32
        pool4 = self.pool(down4)    #128, 16, 16

        middle = self.middle_conv(pool4)  #256, 16, 16

        atrous1 = self.at_conv1(middle)
        atrous2 = self.at_conv1(atrous1)
        atrous3 = self.at_conv1(atrous2)
        atrous4 = self.at_conv1(atrous3)

        up1 = self.deconv1(atrous4)                  #128, 32, 32
        concat1 = torch.cat([down4, up1], dim=1)    #256, 32, 32
        conv1 = self.upconv1(concat1)               #128, 32, 32
        up2 = self.deconv2(conv1)                   #64, 64, 64
        concat2 = torch.cat([down3, up2], dim=1)    #128, 64, 64
        conv2 = self.upconv2(concat2)               #64, 64, 64
        up3 = self.deconv3(conv2)                   #32, 128, 128
        concat3 = torch.cat([down2, up3], dim=1)    #64, 128, 128
        conv3 = self.upconv3(concat3)               #32, 128, 128
        up4 = self.deconv4(conv3)                   #16, 256, 256
        concat4 = torch.cat([down1, up4], dim=1)    #32, 256, 256
        conv4 = self.upconv4(concat4)               #16, 256, 256

        out = self.out(conv4)                       #3, 256, 256

        return out


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
