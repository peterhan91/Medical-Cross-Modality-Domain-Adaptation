import math
import torch
import torch.nn as nn
from torch.autograd import Variable

def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=2, mode='zeros'):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation,
                     padding_mode=mode)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, mode='zeros'):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation,
                     padding_mode=mode)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class R_Block(nn.Module): # basic residual block with dropout layers
    
    def __init__(self, inplanes, planes, DP_rate, stride=1, shortcut_conv=False, groups=1, 
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
        self.dropout1 = nn.Dropout(p=DP_rate)
        self.relu = relu
        self.conv2 = conv3x3(planes, planes)
        self.dropout2 = nn.Dropout(p=DP_rate)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.shortcut_pad = shortcut_conv

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.dropout1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.dropout2(out)
        out = self.bn2(out)

        if self.shortcut_pad:
            padding = Variable(torch.zeros(x.shape[0], 
                                           int(x.shape[1]/2), 
                                           x.shape[2], 
                                           x.shape[3],
                                           dtype=x.dtype, 
                                           device=x.device))
            identity = torch.cat((padding, identity, padding), 1)

        out += identity
        out = self.relu(out)
        return out

class D_Block(nn.Module): # Dilated residual block

    def __init__(self, inplanes, planes, DP_rate, stride=1, shortcut_conv=False, groups=1, 
            dilation=2, norm_layer=None, leaky=False):
        super(D_Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if leaky:
            relu = nn.LeakyReLU(0.2)
        else:
            relu = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(inplanes, planes, stride, groups, dilation)
        self.bn1 = norm_layer(planes)
        self.dropout1 = nn.Dropout(p=DP_rate)
        self.relu = relu
        self.conv2 = conv3x3(planes, planes, stride, groups, dilation)
        self.bn2 = norm_layer(planes)
        self.dropout2 = nn.Dropout(p=DP_rate)
        self.stride = stride
        self.shortcut_pad = shortcut_conv

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.dropout1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.dropout2(out)
        out = self.bn2(out)

        if self.shortcut_pad:
            padding = Variable(torch.zeros(x.shape[0], 
                                           int(x.shape[1]/2), 
                                           x.shape[2], 
                                           x.shape[3],
                                           dtype=x.dtype, 
                                           device=x.device))
            identity = torch.cat((padding, identity, padding), 1)

        out += identity
        out = self.relu(out)
        return out

class P_Block(nn.Module): # Upsampling block with pixel shuffle and 4 conv

    def __init__(self, inplanes, DP_rate, stride=1,
                dilation=2, norm_layer=None, n_cls=5):
        super(P_Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, inplanes, stride)
        self.bn1 = norm_layer(inplanes)
        self.dropout1 = nn.Dropout(p=DP_rate)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = conv3x3(inplanes, inplanes, stride)
        self.bn2 = norm_layer(inplanes)
        self.dropout2 = nn.Dropout(p=DP_rate)
        self.relu = nn.LeakyReLU(0.2)
        self.conv3 = conv3x3(inplanes, inplanes*n_cls, stride, mode='replicate')
        self.dropout3 = nn.Dropout(p=DP_rate)
        self.pixelshuffle = nn.PixelShuffle(8)
        self.conv4 = conv5x5(int(inplanes/64*n_cls), n_cls, stride, mode='replicate')
        # self.conv4 = conv1x1(int(inplanes/64*n_cls), n_cls)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.dropout1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.dropout2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.dropout3(out)
        
        out = self.pixelshuffle(out) 
        out = self.conv4(out)
        return out