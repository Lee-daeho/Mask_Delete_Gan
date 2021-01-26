import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import nn

from PIL import Image

import os

import numpy as np


class editEncoder(nn.Module):
    def __init__(self):
        super(editEncoder, self).__init__()

        self.conv1 = self.doubleConv(4,16)      #downsample
        self.conv2 = self.doubleConv(16,32)
        self.conv3 = self.doubleConv(32,64)
        self.conv4 = self.doubleConv(64,128)

        self.middle_conv = self.doubleConv(128,256) #middle

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