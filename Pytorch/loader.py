import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

digits = re.compile(r'(\d+)')
def tokenize(filename):
    return tuple(int(token) if match else token
                for token, match in
                ((fragment, digits.search(fragment))
                for fragment in digits.split(filename)))

class NumpyDataset(Dataset):
    def __init__(self, directory, modality='MR', mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        self.folder = os.path.join(directory, modality+'_'+mode)  # get the directory of the specified split
        self.Numpylists = os.listdir(os.path.join(self.folder, 'data'))
        self.Numpylists.sort(key=tokenize) 

    def __len__(self):
        return len(self.Numpylists)

    def _label_decomp(label_vol, num_cls=5):
        _batch_shape = list(label_vol.shape)
        _vol = np.zeros(_batch_shape)
        _vol[label_vol == 0] = 1
        _vol = _vol[..., np.newaxis] # the shape will expand to [256, 256, 1, 5]
        for i in range(num_cls):
            if i == 0:
                continue
            _n_slice = np.zeros(label_vol.shape)
            _n_slice[label_vol == i] = 1
            _vol = np.concatenate((_vol, _n_slice[..., np.newaxis]), axis = 3)
        return np.float32(_vol)

    def __getitem__(self, index):
        name_d = os.path.join(self.folder, 'data', self.Numpylists[index])
        name_l = os.path.join(self.folder, 'label', self.Numpylists[index])
        d_ = np.float32(np.load(name_d)) # shape: [256,256,3]
        l_ = np.float32(np.load(name_l)) # shape: [256,256,1]
        l_ = _label_decomp(l_)           # shape: [256,256,1,5]
        d = np.moveaxis(d_, -1, 0)       # shape: [3,256,256]
        l = np.moveaxis(l_, -2, 0)       # shape: [1,256,256,5]
        l = np.moveaxis(l, -1, 0)        # shape: [5,1,256,256]

        sample = {'buffers': d, 'labels': l}

        if self.transform:
            sample = self.transform(sample)

        return sample 

class ToTensor(object):
    def __call__(self, sample):
        buffers, labels = sample['buffers'], sample['labels']
        return {'buffers': torch.from_numpy(buffers),
                'labels': torch.from_numpy(labels)}