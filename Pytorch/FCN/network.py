import torch
import torch.nn as nn
from blocks import *

class Dilated_FCN(nn.Module):
    def __init__(self, input_channel=3, feature_base=16, n_class=5):
        super(Dilated_FCN, self).__init__()
        # Downsampling blocks
        self.conv1 = conv3x3(input_channel, feature_base)
        self.dropout1 = nn.Dropout(p=0.25)
        self.block1 = R_Block(feature_base, feature_base, leaky=True)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.block2 = R_Block(feature_base, feature_base*2, leaky=True)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.block3_1 = R_Block(feature_base*2, feature_base*4, leaky=True)
        self.block3_2 = R_Block(feature_base*4, feature_base*4, leaky=True)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        # residual stacks
        self.block4_1 = R_Block(feature_base*4, feature_base*8, leaky=True)
        self.block4_2 = R_Block(feature_base*8, feature_base*8, leaky=True)
        self.block5_1 = R_Block(feature_base*8, feature_base*16, leaky=True)
        self.block5_2 = R_Block(feature_base*16, feature_base*16, leaky=True)
        self.block6_1 = R_Block(feature_base*16, feature_base*16, leaky=True)
        self.block6_2 = R_Block(feature_base*16, feature_base*16, leaky=True)
        self.block7_1 = R_Block(feature_base*16, feature_base*32, leaky=True)
        self.block7_2 = R_Block(feature_base*32, feature_base*32, leaky=True)
        # Dilated blocks
        self.block8_1 = D_block(feature_base*32, feature_base*32, leaky=True)
        self.block8_2 = D_block(feature_base*32, feature_base*32, leaky=True)
        # Pixel shuffel blocks
        self.block9_1 = P_block(feature_base*32, leaky=True, n_cls=n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.maxpool3(x)

        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.block5_1(x)
        x = self.block5_2(x)
        x = self.block6_1(x)
        x = self.block6_2(x)
        x = self.block7_1(x)
        x = self.block7_2(x)
        x = self.block8_1(x)
        x = self.block8_2(x)
        logits = self.block9_1(x)

        return logits