import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import matplotlib.image as mpimg

import numpy as np

# this is a simple block in our residual model
class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1, stride=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn1(self.conv1(x)))

# read in single file
PATH = "DATA/train/MildDemented/mildDem0.jpg"
single_file = mpimg.imread(PATH)
single_file = single_file[None, None, :, :]

# model
model = Block()
output_tensor = model.forward(Tensor(single_file))
print(output_tensor.shape)

""" resize images potetially useful

x = np.resize(single_file, (1, 1, 128, 128))
print(x.shape)
"""
