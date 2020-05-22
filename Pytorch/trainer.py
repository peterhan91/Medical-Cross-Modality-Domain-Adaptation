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
from util.utils import save_checkpoint
from util.attack import *
from util.dataset import *
from util.dice import *

def train_net(model, device, num_classes, 
                directory, LR, SGD=False, batch_size=8,
                num_epochs=500, save=True, path='fcn.pth.tar'):
    # creating dataloader 
    npdataset = NumpyDataset(directory, transform=transforms.Compose([ToTensor()])) 
    train_dataloader = DataLoader(
                                NumpyDataset(directory, 
                                            transform=transforms.Compose([
                                                HorizontalFlip(0.5),
                                                VerticallFlip(0.5),
                                                RandomRotation(0.5),
                                                RandomElastic(0.8),
                                                # RandomGamma(),
                                                RandomCrop(128),
                                                Normalize(),
                                                ToTensor()
                                                ])), 
                                batch_size=batch_size, shuffle=True, num_workers=48
                                )
    val_dataloader = DataLoader(
                                NumpyDataset(directory, mode='valid', 
                                            transform=transforms.Compose([Normalize(), ToTensor()])), 
                                batch_size=batch_size, shuffle=False, num_workers=48
                                )                                
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    writer = SummaryWriter(comment=f'LR_{LR}_BS_{batch_size}')
    train_step, val_step = 0, 0
    logging.info(f'''Starting training:
        Epochs:          {num_epochs}
        Batch size:      {batch_size}
        Learning rate:   {LR}
        Training size:   {dataset_sizes['train']}
        Validation size: {dataset_sizes['val']}
        Device:          {device.type}
    ''')
    if SGD:
        optimizer = optim.SGD(model.parameters(), lr=LR, 
                                momentum=0.9, weight_decay=1e-4) 
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-6)

    # training preparation
    start = time.time()
    epoch_resume = 0
    best_loss = 999
    criterion_d = DiceLoss()

    if os.path.exists(path):
        checkpoint = torch.load(path)
        logging.info("Reloading model from previously saved checkpoint")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        epoch_resume = checkpoint["epoch"]

    # start training
    for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", 
                        initial=epoch_resume, total=num_epochs, ascii=True):
        for phase in ['train', 'val']:
            running_loss, running_loss_c, running_loss_d = 0.0, 0.0, 0.0
            epoch_loss = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for sample in dataloaders[phase]:
                inputs, labels, labels_ = sample['buffers'], sample['labels'], sample['labels_']
                inputs = inputs.to(device, dtype= torch.float)  # (8, 3, 256, 256)
                labels = labels.to(device, dtype= torch.float)   # (8, 5, 256, 256)
                labels_ = labels_.to(device, dtype= torch.long) # (8, 1, 256, 256)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    logits = model(inputs) 
                    dims = (0,) + tuple(range(2, labels.ndimension())) # (0, 2, 3)
                    weights = 1.0 - torch.sum(labels, dims)/torch.sum(labels)
                    loss_c = nn.CrossEntropyLoss()(logits, labels_)
                    loss_d = criterion_d(logits, labels)
                    loss = loss_d + loss_c
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()  

                running_loss += loss.item() * inputs.size(0)
                running_loss_c += loss_c.item() * inputs.size(0)
                running_loss_d += loss_d.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss_c = running_loss_c / dataset_sizes[phase]
            epoch_loss_d = running_loss_d / dataset_sizes[phase]
            logging.info(f"{phase} Loss: {epoch_loss} CrossEntropyLoss: {epoch_loss_c} DiceLoss: {epoch_loss_d}")
            
            if phase == 'train':
                writer.add_scalar('Loss/'+phase, epoch_loss, train_step)
                writer.add_scalar('CrossEntropyLoss/'+phase, epoch_loss_c, train_step)
                writer.add_scalar('DiceLoss/'+phase, epoch_loss_d, train_step)
                train_step += 1
            
            if phase == 'val':
                # scheduler.step()
                scheduler.step(epoch_loss)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), val_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), val_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], val_step)
                writer.add_scalar('Loss/'+phase, epoch_loss, val_step)
                writer.add_scalar('CrossEntropyLoss/'+phase, epoch_loss_c, val_step)
                writer.add_scalar('DiceLoss/'+phase, epoch_loss_d, val_step)
                val_step += 1
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_checkpoint(model, optimizer, epoch, path)    

    writer.close()
    time_elapsed = time.time() - start    
    logging.info(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = UNet(n_channels=3, n_classes=5, n_features=32)
    model = Dilated_FCN(feature_base=16, DP_rate=0.0)
    logging.info(f'Using device {device}')
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{model.n_features} basic feature channels')
    model.to(device=device)
    checkpoint_dir = os.path.join('./checkpoint/FCN/', str(datetime.datetime.now().time()))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    try:
        train_net(model=model,
                  device=device,
                  num_classes=5,
                  directory='/media/tianyu.han/mri-scratch/DeepLearning/Cardiac_4D/MRCT/',
                  LR=3e-4,
                  batch_size=128,
                  num_epochs=100,
                  path=os.path.join(checkpoint_dir,
                                    'fcn.pth.tar')
                  )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 
                                                    'INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)