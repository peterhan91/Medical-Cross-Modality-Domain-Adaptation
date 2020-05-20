import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_features, bilinear=True):
        super(UNet, self).__init__()
        self.n_features = n_features
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, n_features, dilate=False)
        self.down1 = Down(n_features, n_features*2, dilate=False)
        self.down2 = Down(n_features*2, n_features*4, dilate=False)
        self.down3 = BNDilateConv(n_features*4, n_features*8 // factor)
        # self.down3 = Down(n_features*4, n_features*8 // factor)
        # self.down4 = Down(n_features*8, n_features*16 // factor)
        # self.up1 = Up(n_features*16, n_features*8 // factor, bilinear)
        self.up2 = Up(n_features*8, n_features*4 // factor, bilinear)
        self.up3 = Up(n_features*4, n_features*2 // factor, bilinear)
        self.up4 = Up(n_features*2, n_features, bilinear)
        self.outc = OutConv(n_features, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits