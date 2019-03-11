import torch
import torch.nn as nn


class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride, padding, downsample=None):
        self._mid_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1, 
                               bias=False)

        self.conv2 = nn.Conv2d(self._mid_channels, 
                               self._mid_channels,
                               kernel_size=3, 
                               stride=stride, 
                               padding=1,
                               bias=False)

        self.conv3 = nn.Conv2d(self._mid_channels,
                               out_channels,
                               kernel_size=1,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(self._mid_channels)
        self.bn2 = nn.BatchNorm2d(self._mid_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)



        return out


