import os
import json
import logging
import numpy as np
from sklearn.metrics import accuracy_score
import torch

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda().float()

def save_checkpoint(model, optimizer, epoch, path):
    print('Saving Model')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'opt_dict': optimizer.state_dict(),
        }, path)