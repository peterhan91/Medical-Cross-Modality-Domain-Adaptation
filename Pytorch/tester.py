import os
import time
import numpy as np
from tqdm import tqdm
import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchsummary import summary
from torchvision import transforms, utils

from FCN.network import Dilated_FCN
from util.utils import save_checkpoint
from util.attack import *
from util.dataset import NumpyDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

model = Dilated_FCN().to(device)
summary(model, input_size=(3, 256, 256))