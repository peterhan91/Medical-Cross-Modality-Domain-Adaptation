import os
import time
import datetime
import numpy as np
from tqdm import tqdm
import logging
import copy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

from FCN.network import Dilated_FCN
from UNet.unet import UNet

def eval_img(model, device, path, outdir):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        logging.info("Reloading model from previously saved checkpoint")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        epoch_resume = checkpoint["epoch"]
    else:
        print('no models found!')
        break
    val_dataloader = DataLoader(
                            NumpyDataset(directory, mode='valid', 
                                        transform=transforms.Compose([ToTensor()])), 
                            batch_size=1, num_workers=32
                            ) 
    for i_batch, sample in enumerate(val_dataloader):
        inputs, labels, labels_ = sample['buffers'], sample['labels'], sample['labels_']
        inputs = inputs.to(device, dtype= torch.float)  # (1, 3, 256, 256)
        labels = labels.to(device, dtype= torch.float)   # (1, 5, 256, 256)
        labels_ = labels_.to(device, dtype= torch.long) # (1, 1, 256, 256)

        logits = model(inputs) 
        preds = get_segmaps(logits)
        fig, axs = plt.subplots(1, 3, facecolor='w', edgecolor='k')
        axs[0].imshow(inputs)

