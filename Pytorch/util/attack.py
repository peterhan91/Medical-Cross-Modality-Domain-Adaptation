import sys
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import tensor2cuda

def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon
        x = torch.max(torch.min(x, max_x), min_x)

    elif _type == 'l2':
        dist = (x - original_x)
        dist = dist.view(x.shape[0], -1)
        dist_norm = torch.norm(dist, dim=1, keepdim=True)
        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
        # dist = F.normalize(dist, p=2, dim=1)
        dist = dist / dist_norm
        dist *= epsilon
        dist = dist.view(x.shape)
        x = (original_x + dist) * mask.float() + x * (1 - mask.float())

    else:
        raise NotImplementedError

    return x

class FastGradientSignUntargeted():
    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, _type='linf'):
        self.model = model
        self.model.eval()
        self.epsilon = epsilon
        self.alpha = alpha
        self.min_val = min_val
        self.max_val = max_val
        self.max_iters = max_iters
        self._type = _type
        
    def perturb(self, original_images, labels, reduction4loss='mean', random_start=False):
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = tensor2cuda(rand_perturb)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        
        else:
            x = original_images.clone()

        x.requires_grad = True 
        self.model.eval()
        with torch.enable_grad():
            for _iter in range(self.max_iters):
                outputs = self.model(x)
                loss = F.binary_cross_entropy(F.sigmoid(outputs), labels, reduction=reduction4loss)
                if reduction4loss == 'none':
                    grad_outputs = tensor2cuda(torch.ones(loss.shape))
                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, 
                        only_inputs=True)[0]

                x.data += self.alpha * torch.sign(grads.data) 
                x = project(x, original_images, self.epsilon, self._type)
                x.clamp_(self.min_val, self.max_val)
        self.model.train()
        
        return x
