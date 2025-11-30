import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from numpy import genfromtxt


class InceptionBlock1a(nn.Module):
    """Implementation of an inception block 1a"""

    def __init__(self):
        super(InceptionBlock1a, self).__init__()

        # 3x3 branch
        self.conv_3x3_1 = nn.Conv2d(192, 96, kernel_size=1, stride=1)
        self.bn_3x3_1 = nn.BatchNorm2d(96, eps=0.00001)
        self.conv_3x3_2 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.bn_3x3_2 = nn.BatchNorm2d(128, eps=0.00001)

        # 5x5 branch
        self.conv_5x5_1 = nn.Conv2d(192, 16, kernel_size=1, stride=1)
        self.bn_5x5_1 = nn.BatchNorm2d(16, eps=0.00001)
        self.conv_5x5_2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn_5x5_2 = nn.BatchNorm2d(32, eps=0.00001)

        # Pool branch
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv_pool = nn.Conv2d(192, 32, kernel_size=1, stride=1)
        self.bn_pool = nn.BatchNorm2d(32, eps=0.00001)

        # 1x1 branch
        self.conv_1x1 = nn.Conv2d(192, 64, kernel_size=1, stride=1)
        self.bn_1x1 = nn.BatchNorm2d(64, eps=0.00001)

    def forward(self, x):
        # 3x3 branch
        x_3x3 = F.relu(self.bn_3x3_1(self.conv_3x3_1(x)))
        x_3x3 = F.relu(self.bn_3x3_2(self.conv_3x3_2(x_3x3)))

        # 5x5 branch
        x_5x5 = F.relu(self.bn_5x5_1(self.conv_5x5_1(x)))
        x_5x5 = F.relu(self.bn_5x5_2(self.conv_5x5_2(x_5x5)))

        # Pool branch
        x_pool = self.maxpool(x)
        x_pool = F.relu(self.bn_pool(self.conv_pool(x_pool)))
        # Zero padding to match dimensions
        x_pool = F.pad(x_pool, (3, 4, 3, 4))

        # 1x1 branch
        x_1x1 = F.relu(self.bn_1x1(self.conv_1x1(x)))

        # Concatenate
        return torch.cat([x_3x3, x_5x5, x_pool, x_1x1], dim=1)


class InceptionBlock1b(nn.Module):
    """Implementation of an inception block 1b"""

    def __init__(self):
        super(InceptionBlock1b, self).__init__()

        # 3x3 branch
        self.conv_3x3_1 = nn.Conv2d(256, 96, kernel_size=1, stride=1)
        self.bn_3x3_1 = nn.BatchNorm2d(96, eps=0.00001)
        self.conv_3x3_2 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.bn_3x3_2 = nn.BatchNorm2d(128, eps=0.00001)

        # 5x5 branch
        self.conv_5x5_1 = nn.Conv2d(256, 32, kernel_size=1, stride=1)
        self.bn_5x5_1 = nn.BatchNorm2d(32, eps=0.00001)
        self.conv_5x5_2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn_5x5_2 = nn.BatchNorm2d(64, eps=0.00001)

        # Pool branch
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=3)
        self.conv_pool = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        self.bn_pool = nn.BatchNorm2d(64, eps=0.00001)

        # 1x1 branch
        self.conv_1x1 = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        self.bn_1x1 = nn.BatchNorm2d(64, eps=0.00001)

    def forward(self, x):
        # 3x3 branch
        x_3x3 = F.relu(self.bn_3x3_1(self.conv_3x3_1(x)))
        x_3x3 = F.relu(self.bn_3x3_2(self.conv_3x3_2(x_3x3)))

        # 5x5 branch
        x_5x5 = F.relu(self.bn_5x5_1(self.conv_5x5_1(x)))
        x_5x5 = F.relu(self.bn_5x5_2(self.conv_5x5_2(x_5x5)))

        # Pool branch
        x_pool = self.avgpool(x)
        x_pool = F.relu(self.bn_pool(self.conv_pool(x_pool)))
        # Zero padding to match dimensions
        x_pool = F.pad(x_pool, (4, 4, 4, 4))

        # 1x1 branch
        x_1x1 = F.relu(self.bn_1x1(self.conv_1x1(x)))

        # Concatenate
        return torch.cat([x_3x3, x_5x5, x_pool, x_1x1], dim=1)


class Conv2dBN(nn.Module):
    """Helper module for Conv2d + BatchNorm + ReLU with optional padding"""

    def __init__(self, in_channels, out_channels1, kernel_size1=(1, 1), stride1=(1, 1),
                 out_channels2=None, kernel_size2=(3, 3), stride2=(1, 1), padding=None):
        super(Conv2dBN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=kernel_size1, stride=stride1)
        self.bn1 = nn.BatchNorm2d(out_channels1, eps=0.00001)

        self.padding = padding
        self.has_second_conv = out_channels2 is not None

        if self.has_second_conv:
            self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=kernel_size2, stride=stride2)
            self.bn2 = nn.BatchNorm2d(out_channels2, eps=0.00001)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        if self.padding is not None:
            if isinstance(self.padding, tuple) and len(self.padding) == 2:
                # padding is (pad_h, pad_w)
                x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
            else:
                x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        if self.has_second_conv:
            x = F.relu(self.bn2(self.conv2(x)))

        return x


class InceptionBlock1c(nn.Module):
    """Implementation of an inception block 1c"""

    def __init__(self):
        super(InceptionBlock1c, self).__init__()

        # 3x3 branch
        self.conv_3x3 = Conv2dBN(256, 128, (1, 1), (1, 1), 256, (3, 3), (2, 2), (1, 1))

        # 5x5 branch
        self.conv_5x5 = Conv2dBN(256, 32, (1, 1), (1, 1), 64, (5, 5), (2, 2), (2, 2))

        # Pool branch
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        # 3x3 branch
        x_3x3 = self.conv_3x3(x)

        # 5x5 branch
        x_5x5 = self.conv_5x5(x)

        # Pool branch
        x_pool = self.maxpool(x)
        x_pool = F.pad(x_pool, (0, 1, 0, 1))

        # Concatenate
        return torch.cat([x_3x3, x_5x5, x_pool], dim=1)


class InceptionBlock2a(nn.Module):
    """Implementation of an inception block 2a"""

    def __init__(self):
        super(InceptionBlock2a, self).__init__()

        # 3x3 branch
        self.conv_3x3 = Conv2dBN(640, 96, (1, 1), (1, 1), 192, (3, 3), (1, 1), (1, 1))

        # 5x5 branch
        self.conv_5x5 = Conv2dBN(640, 32, (1, 1), (1, 1), 64, (5, 5), (1, 1), (2, 2))

        # Pool branch
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=3)
        self.conv_pool = Conv2dBN(640, 128, (1, 1), (1, 1), padding=(2, 2))

        # 1x1 branch
        self.conv_1x1 = Conv2dBN(640, 256, (1, 1), (1, 1))

    def forward(self, x):
        # 3x3 branch
        x_3x3 = self.conv_3x3(x)

        # 5x5 branch
        x_5x5 = self.conv_5x5(x)

        # Pool branch
        x_pool = self.avgpool(x)
        x_pool = self.conv_pool(x_pool)

        # 1x1 branch
        x_1x1 = self.conv_1x1(x)

        # Concatenate
        return torch.cat([x_3x3, x_5x5, x_pool, x_1x1], dim=1)


class InceptionBlock2b(nn.Module):
    """Implementation of an inception block 2b (inception4e)"""

    def __init__(self):
        super(InceptionBlock2b, self).__init__()

        # 3x3 branch
        self.conv_3x3 = Conv2dBN(640, 160, (1, 1), (1, 1), 256, (3, 3), (2, 2), (1, 1))

        # 5x5 branch
        self.conv_5x5 = Conv2dBN(640, 64, (1, 1), (1, 1), 128, (5, 5), (2, 2), (2, 2))

        # Pool branch
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        # 3x3 branch
        x_3x3 = self.conv_3x3(x)

        # 5x5 branch
        x_5x5 = self.conv_5x5(x)

        # Pool branch
        x_pool = self.maxpool(x)
        x_pool = F.pad(x_pool, (0, 1, 0, 1))

        # Concatenate
        return torch.cat([x_3x3, x_5x5, x_pool], dim=1)


class InceptionBlock3a(nn.Module):
    """Implementation of an inception block 3a"""

    def __init__(self):
        super(InceptionBlock3a, self).__init__()

        # 3x3 branch
        self.conv_3x3 = Conv2dBN(1024, 96, (1, 1), (1, 1), 384, (3, 3), (1, 1), (1, 1))

        # Pool branch
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=3)
        self.conv_pool = Conv2dBN(1024, 96, (1, 1), (1, 1), padding=(1, 1))

        # 1x1 branch
        self.conv_1x1 = Conv2dBN(1024, 256, (1, 1), (1, 1))

    def forward(self, x):
        # 3x3 branch
        x_3x3 = self.conv_3x3(x)

        # Pool branch
        x_pool = self.avgpool(x)
        x_pool = self.conv_pool(x_pool)

        # 1x1 branch
        x_1x1 = self.conv_1x1(x)

        # Concatenate
        return torch.cat([x_3x3, x_pool, x_1x1], dim=1)


class InceptionBlock3b(nn.Module):
    """Implementation of an inception block 3b"""

    def __init__(self):
        super(InceptionBlock3b, self).__init__()

        # 3x3 branch
        self.conv_3x3 = Conv2dBN(736, 96, (1, 1), (1, 1), 384, (3, 3), (1, 1), (1, 1))

        # Pool branch
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv_pool = Conv2dBN(736, 96, (1, 1), (1, 1))

        # 1x1 branch
        self.conv_1x1 = Conv2dBN(736, 256, (1, 1), (1, 1))

    def forward(self, x):
        # 3x3 branch
        x_3x3 = self.conv_3x3(x)

        # Pool branch
        x_pool = self.maxpool(x)
        x_pool = self.conv_pool(x_pool)
        x_pool = F.pad(x_pool, (1, 1, 1, 1))

        # 1x1 branch
        x_1x1 = self.conv_1x1(x)

        # Concatenate
        return torch.cat([x_3x3, x_pool, x_1x1], dim=1)


class FaceRecoModel(nn.Module):
    """
    Implementation of the Inception model used for FaceNet

    Arguments:
    input_shape -- shape of the images of the dataset (channels, height, width)

    Returns:
    model -- a PyTorch model instance
    """

    def __init__(self, input_shape=(3, 96, 96)):
        super(FaceRecoModel, self).__init__()

        # First Block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Second Block
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64, eps=0.00001)

        # Third Block
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(192, eps=0.00001)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Inception blocks
        self.inception_1a = InceptionBlock1a()
        self.inception_1b = InceptionBlock1b()
        self.inception_1c = InceptionBlock1c()

        self.inception_2a = InceptionBlock2a()
        self.inception_2b = InceptionBlock2b()

        self.inception_3a = InceptionBlock3a()
        self.inception_3b = InceptionBlock3b()

        # Top layer
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(736, 128)

    def forward(self, x):
        # First Block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        # Second Block
        x = F.relu(self.bn2(self.conv2(x)))

        # Third Block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool2(x)

        # Inception blocks
        x = self.inception_1a(x)
        x = self.inception_1b(x)
        x = self.inception_1c(x)

        x = self.inception_2a(x)
        x = self.inception_2b(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)

        # Top layer
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        # L2 normalization
        x = F.normalize(x, p=2, dim=1)

        return x


def faceRecoModel(input_shape=(3, 96, 96)):
    """
    Creates and returns the FaceNet Inception model

    Arguments:
    input_shape -- shape of the images (channels, height, width)

    Returns:
    model -- PyTorch FaceRecoModel instance
    """
    return FaceRecoModel(input_shape)
