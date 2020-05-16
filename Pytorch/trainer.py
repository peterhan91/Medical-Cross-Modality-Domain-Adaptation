import os
import time
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
from util.utils import save_checkpoint
from util.attack import *
from util.dataset import NumpyDataset, ToTensor
from util.dice import dice_loss

def train_net(model, device, num_classes, directory, LR=0.2, batch_size=8,
            num_epochs=500, save=True, path='fcn.pth.tar'):
    # creating dataloader 
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

    writer = SummaryWriter(comment=f'LR_{LR}_BS_{batch_size}')
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {num_epochs}
        Batch size:      {batch_size}
        Learning rate:   {LR}
        Training size:   {dataset_sizes['train']}
        Validation size: {dataset_sizes['val']}
        Device:          {device.type}
    ''')

    optimizer = optim.SGD(model.parameters(), lr=LR, 
                            momentum=0.2, weight_decay=1e-4)  
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    # training preparation
    start = time.time()
    epoch_resume = 0
    best_loss = 1e6

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
                labels = labels.to(device, dtype= torch.long)   # (8, 5, 256, 256)
                labels_ = labels_.to(device, dtype= torch.long) # (8, 1, 256, 256)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    logits = model(inputs) 
                    dims = (0,) + tuple(range(2, labels.ndimension())) # (0, 2, 3)
                    weights = 1.0 - torch.sum(labels, dims)/torch.sum(labels)
                    loss_c = nn.CrossEntropyLoss(weight=weights.float())(logits, labels_)
                    loss_d = dice_loss(logits, labels)
                    loss = loss_c + loss_d
                    writer.add_scalar('Loss/'+phase, loss.item(), global_step)
                    writer.add_scalar('CrossEntropyLoss/'+phase, loss_c.item(), global_step)
                    writer.add_scalar('DiceLoss/'+phase, loss_d.item(), global_step)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   

                running_loss += loss.item() * inputs.size(0)
                running_loss_c += loss_c.item() * inputs.size(0)
                running_loss_d += loss_d.item() * inputs.size(0)
                global_step += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss_c = running_loss_c / dataset_sizes[phase]
            epoch_loss_d = running_loss_d / dataset_sizes[phase]
            logging.info(f"{phase} Loss: {epoch_loss} CrossEntropyLoss: {epoch_loss_c} DiceLoss: {epoch_loss_d}")
            
            if phase == 'val':
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    save_checkpoint(model, optimizer, epoch, path)    

    writer.close()
    time_elapsed = time.time() - start    
    logging.info(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Dilated_FCN()
    logging.info(f'Using device {device}')
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{model.n_features} basic feature channels')
    model.to(device=device)
    checkpoint_dir = './checkpoint/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    try:
        train_net(model=model,
                  device=device,
                  num_classes=5,
                  directory='/media/tianyu.han/mri-scratch/DeepLearning/Cardiac_4D/MRCT/',
                  LR=0.2,
                  batch_size=8,
                  num_epochs=100,
                  path=os.path.join(checkpoint_dir, 'fcn.pth.tar')
                  )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir,'INTERRUPTED.pth'))
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)