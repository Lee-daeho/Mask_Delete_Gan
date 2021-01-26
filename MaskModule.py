import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import nn

import os


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.conv1 = self.doubleConv(3,16)      #downsample
        self.conv2 = self.doubleConv(16,32)
        self.conv3 = self.doubleConv(32,64)
        self.conv4 = self.doubleConv(64,128)

        self.middle_conv = self.doubleConv(128,256) #middle

        self.deconv1 = self.deconv(256,128) #upsample
        self.deconv2 = self.deconv(128,64)
        self.deconv3 = self.deconv(64,32)
        self.deconv4 = self.deconv(32,16)

        self.out = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
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

        up1 = self.deconv1(middle)                  #128, 32, 32
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