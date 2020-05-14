import math
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class R_Block(nn.Module):
    
    def __init__(self, inplanes, planes, stride=1, shortcut=False, groups=1, base_width=16, 
                dilation=1, norm_layer=None, leaky=False):
        super(R_Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if leaky:
            relu = nn.LeakyReLU(0.2)
        else:
            relu = nn.ReLU(inplace=True)
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.dropout1 = nn.Dropout(p=0.25)
        self.relu = relu
        self.conv2 = conv3x3(planes, planes)
        self.dropout2 = nn.Dropout(p=0.25)
        self.bn2 = norm_layer(planes)
        if not shortcut:
            self.downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes)
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.dropout1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.dropout2(x)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out