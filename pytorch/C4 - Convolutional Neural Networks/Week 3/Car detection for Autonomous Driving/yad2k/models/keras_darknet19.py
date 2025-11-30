"""Darknet19 Model Defined in PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DarknetConv2d_BN_Leaky(nn.Module):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DarknetConv2d_BN_Leaky, self).__init__()

        # Ensure 'same' padding
        if padding == 'same':
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

        # Apply weight regularization by setting weight_decay in optimizer
        # L2 regularization with lambda=5e-4

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class BottleneckBlock(nn.Module):
    """Bottleneck block of 3x3, 1x1, 3x3 convolutions."""

    def __init__(self, in_channels, outer_filters, bottleneck_filters):
        super(BottleneckBlock, self).__init__()

        self.conv1 = DarknetConv2d_BN_Leaky(in_channels, outer_filters, 3, padding='same')
        self.conv2 = DarknetConv2d_BN_Leaky(outer_filters, bottleneck_filters, 1, padding='same')
        self.conv3 = DarknetConv2d_BN_Leaky(bottleneck_filters, outer_filters, 3, padding='same')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class BottleneckX2Block(nn.Module):
    """Bottleneck block of 3x3, 1x1, 3x3, 1x1, 3x3 convolutions."""

    def __init__(self, in_channels, outer_filters, bottleneck_filters):
        super(BottleneckX2Block, self).__init__()

        self.bottleneck = BottleneckBlock(in_channels, outer_filters, bottleneck_filters)
        self.conv4 = DarknetConv2d_BN_Leaky(outer_filters, bottleneck_filters, 1, padding='same')
        self.conv5 = DarknetConv2d_BN_Leaky(bottleneck_filters, outer_filters, 3, padding='same')

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class DarknetBody(nn.Module):
    """Generate first 18 conv layers of Darknet-19."""

    def __init__(self, in_channels=3):
        super(DarknetBody, self).__init__()

        self.conv1 = DarknetConv2d_BN_Leaky(in_channels, 32, 3, padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = DarknetConv2d_BN_Leaky(32, 64, 3, padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck1 = BottleneckBlock(64, 128, 64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck2 = BottleneckBlock(128, 256, 128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck_x2_1 = BottleneckX2Block(256, 512, 256)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.bottleneck_x2_2 = BottleneckX2Block(512, 1024, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.bottleneck1(x)
        x = self.pool3(x)

        x = self.bottleneck2(x)
        x = self.pool4(x)

        x = self.bottleneck_x2_1(x)
        x = self.pool5(x)

        x = self.bottleneck_x2_2(x)

        return x


class Darknet19(nn.Module):
    """Generate Darknet-19 model for Imagenet classification."""

    def __init__(self, in_channels=3, num_classes=1000):
        super(Darknet19, self).__init__()

        self.body = DarknetBody(in_channels)
        self.classifier = nn.Conv2d(1024, num_classes, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.body(x)
        x = self.classifier(x)
        x = self.softmax(x)
        return x


def darknet_body(in_channels=3):
    """Create and return Darknet body"""
    return DarknetBody(in_channels)


def darknet19(in_channels=3, num_classes=1000):
    """Create and return Darknet-19 model"""
    return Darknet19(in_channels, num_classes)
