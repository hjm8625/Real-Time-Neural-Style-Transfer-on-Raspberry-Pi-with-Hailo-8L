"""Hailo-friendly image transformation network."""

import torch
from torch import nn


# Convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + residual

# Upsample block using nearest-neighbor resize followed by convolution
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# Main network
class StyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(3, 32, kernel_size=9)
        self.encoder2 = ConvBlock(32, 64, kernel_size=3, stride=2)
        self.encoder3 = ConvBlock(64, 128, kernel_size=3, stride=2)
        # Five residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(128) for _ in range(5)]
        )
        self.upsample1 = UpsampleBlock(128, 64)
        self.upsample2 = UpsampleBlock(64, 32)
        self.output_conv = nn.Conv2d(32, 3, kernel_size=9, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.output_conv(x)
        return self.tanh(x)
