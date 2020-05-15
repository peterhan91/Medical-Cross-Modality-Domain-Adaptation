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
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

from FCN.network import Dilated_FCN
from util.utils import save_checkpoint
from util.attack import *
from util.dataset import NumpyDataset, ToTensor

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)


def train(num_classes, directory, LR=0.2, batch_size=8,
        num_epochs=50, save=True, path='mrnet.pth.tar'):

    npdataset = NumpyDataset(directory, transform=transforms.Compose([ToTensor()])) 
    train_dataloader = DataLoader(
                                NumpyDataset(directory, 
                                            transform=transforms.Compose([ToTensor()])), 
                                batch_size=batch_size, shuffle=True, num_workers=8
                                )
    val_dataloader = DataLoader(
                                NumpyDataset(directory, mode='valid', 
                                            transform=transforms.Compose([ToTensor()])), 
                                batch_size=batch_size, num_workers=4
                                )                                
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    SummaryWriter(comment=f'LR_{LR}_BS_{batch_size}')
    model = Dilated_FCN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, 
                        momentum=0.2, weight_decay=1e-4)  
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0
    best_loss = 1e6

    if os.path.exists(path):
        checkpoint = torch.load(path)
        print("Reloading model from previously saved checkpoint")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        epoch_resume = checkpoint["epoch"]

    for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", 
                        initial=epoch_resume, total=num_epochs, ascii=True):
        
        for phase in ['train', 'val']:
            running_loss = 0.0
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for sample in dataloaders[phase]:
                inputs, labels = sample['buffers'], sample['labels']
                inputs = inputs.to(device, dtype= torch.float)
                labels = labels.to(device, dtype= torch.float)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs) 
                    loss = criterion(outputs, labels)

                    # Backpropagate and optimize iff in training mode, else there's no intermediate
                    # values to backpropagate with and will throw an error.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   

                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss}")

            if phase == 'val' and epoch_loss > best_loss:
                patient +=1
                if patient >= MAX_patient and LR > 1e-5:
                    print("decay loss from " + str(LR) + " to " +
                        str(LR * 0.1) + " as no improvement in val loss")
                    LR = LR * 0.1
                    # optimizer = optim.Adam(model.parameters(), lr=LR)
                    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
                    print("created new optimizer with LR " + str(LR))
                    patient = 0
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                save_checkpoint(model, optimizer, epoch, path)
                patient = 0     

    # print the total time needed, HH:MM:SS format
    time_elapsed = time.time() - start    
    print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")
