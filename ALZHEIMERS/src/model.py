import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

""" Hidden EzNet -> Backup

import torch
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
"""

# useful if you want to use it
def dynamic_padding(kernel_size):
    return (kernel_size[0]//2, kernel_size[1]//2)

class Block(nn.Module):
    # model less general than source code in torch docs
    def __init__(self):
        super(Block, self).__init__()

        # layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        """ optional additional layers
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        """

    # forward propagation
    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn1(self.conv1(x)))

class BottleNeck(nn.Module):
    def __init__(self, expansion: int = 4):
        super(BottleNeck, self).__init__()
        # 1 pass
        self.conv1 = nn.Conv2d(32, 176, kernel_size=1, stride=1, bais=False)
        self.bn1 = nn.BatchNorm(32)

        # relu function
        self.relu = nn.ReLU(inplace=True)

        # 2 pass
        self.conv2 = nn.Conv2d(176, 176, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm(176)

        # 3 pass
        self.conv3 = nn.Conv2d(176, 32 * expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm(32 * expansion)

    def forward(self, x: Tensor) -> Tensor:
        # only a 1 pass
        # edit to do 3 pass
        return self.relu(self.bn1(self.conv1(x)))

class ResNet(nn.Module):
    def __init__(self,
                 block: Type[Union[Block, BottleNeck]],
                 layers: List[int],
                 groups: int = 100,
                 base_width: int = 176,
                 in_planes: int = 1) -> None:
        super(ResNet, self).__init__()

        # layers and attributes
        self.groups = groups
        self.base_width = base_width
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # first layer
        self.layer1 = self._make_layer(block, 64, layers[0])

    def _make_layer(self, block: Type[Union[Block, BottleNeck]],
                    planes: int, blocks: int) -> nn.Sequential:
        """ maybe important
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        """
        layers = []
        layers.append(block(1, planes))
        self.inplanes = planes * block.expansion

        # fix this method cause constructors
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)
