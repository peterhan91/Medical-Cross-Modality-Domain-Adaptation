import os
import json
import logging
import numpy as np
from sklearn.metrics import accuracy_score
import torch

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda().float()