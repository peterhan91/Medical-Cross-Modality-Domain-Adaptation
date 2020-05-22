import os
import re
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.transform import rotate
from skimage import exposure
from util.utils import *

digits = re.compile(r'(\d+)')
def tokenize(filename):
    return tuple(int(token) if match else token
                for token, match in
                ((fragment, digits.search(fragment))
                for fragment in digits.split(filename)))

class NumpyDataset(Dataset):
    def __init__(self, directory, modality='MR', n_cls=5,
                    mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        self.folder = os.path.join(directory, modality+'_'+mode)  # get the directory of the specified split
        self.Numpylists = os.listdir(os.path.join(self.folder, 'data'))
        self.Numpylists.sort(key=tokenize)
        self.n_class = n_cls 

    def __len__(self):
        return len(self.Numpylists)

    def _label_decomp(self, label_vol, num_cls):
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
        return np.float32(np.squeeze(_vol))

    def __getitem__(self, index):
        name_d = os.path.join(self.folder, 'data', self.Numpylists[index])
        name_l = os.path.join(self.folder, 'label', self.Numpylists[index])
        d_ = np.float32(np.load(name_d))                # shape: [256,256,3]
        l_ = np.float32(np.load(name_l))                # shape: [256,256,1]
        l = self._label_decomp(l_, self.n_class)        # shape: [256,256,5]
        assert d_.shape == (256, 256, 3)
        assert l.shape == (256, 256, self.n_class)
        assert l_.shape == (256, 256, 1)

        sample = {'buffers': d_, 'labels': l, 'labels_':np.squeeze(l_)}
        if self.transform:
            sample = self.transform(sample)
        
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        buffers, labels, labels_ = sample['buffers'], sample['labels'], sample['labels_']
        buffers = buffers.transpose((2, 0, 1))
        labels = labels.transpose((2, 0, 1))
        return {
                'buffers': torch.from_numpy(buffers).type(torch.FloatTensor),
                'labels': torch.from_numpy(labels).type(torch.FloatTensor),
                'labels_': torch.from_numpy(labels_).type(torch.FloatTensor)
                }

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask, mask_ = sample['buffers'], sample['labels'], sample['labels_']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w, :]
        mask = mask[top: top + new_h,
                      left: left + new_w, :]
        mask_ = mask_[top: top + new_h,
                      left: left + new_w]

        return {'buffers': image, 'labels': mask, 'labels_':mask_}

class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask, mask_ = sample['buffers'], sample['labels'], sample['labels_']

        if np.random.rand() > self.flip_prob:
            return {'buffers': image, 'labels': mask, 'labels_':mask_}

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
        mask_ = np.fliplr(mask_).copy()

        return {'buffers': image, 'labels': mask, 'labels_':mask_}

class VerticallFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask, mask_ = sample['buffers'], sample['labels'], sample['labels_']

        if np.random.rand() > self.flip_prob:
            return {'buffers': image, 'labels': mask, 'labels_':mask_}

        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
        mask_ = np.flipud(mask_).copy()

        return {'buffers': image, 'labels': mask, 'labels_':mask_}

class RandomGamma(object):
    # change Gamma (brightness) of the input image only  

    def __call__(self, sample):
        image, mask, mask_ = sample['buffers'], sample['labels'], sample['labels_']
        gamma = random.uniform(0.8, 1.2)

        for n in range(image.shape[-1]):
            # image[:,:,n] = exposure.rescale_intensity(image[:,:,n])
            image[:,:,n] = exposure.adjust_gamma(image[:,:,n], gamma)

        return {'buffers': image, 'labels': mask, 'labels_':mask_} 

class RandomRotation(object):

    def __init__(self, prob):
        self.prob = prob
        self.rotation = [90, 180, 270]    

    def __call__(self, sample):
        image, mask, mask_ = sample['buffers'], sample['labels'], sample['labels_']
        index = np.random.randint(0, 3)

        if np.random.rand() > self.prob:
            return {'buffers': image, 'labels': mask, 'labels_':mask_}

        for n in range(image.shape[-1]):
            image[:,:,n] = rotate(image[:,:,n], self.rotation[index])
        for m in range(mask.shape[-1]):
            mask[:,:,m] = rotate(mask[:,:,m], self.rotation[index])
        mask_ = rotate(mask_, self.rotation[index])

        return {'buffers': image, 'labels': mask, 'labels_':mask_} 

class RandomElastic(object):

    def __init__(self, prob):
        self.prob = prob  

    def __call__(self, sample):
        image, mask, mask_ = sample['buffers'], sample['labels'], sample['labels_']

        if np.random.rand() > self.prob:
            return {'buffers': image, 'labels': mask, 'labels_':mask_}

        im_merge = np.concatenate((image, mask, mask_[...,None]), axis=2)
        im_merge_t = elastic_transform(im_merge, 
                                    im_merge.shape[1] * 3, 
                                    im_merge.shape[1] * 0.1,
                                    im_merge.shape[1] * 0.1)
        
        image = im_merge_t[..., :image.shape[-1]]
        mask = im_merge_t[..., image.shape[-1]:image.shape[-1]+mask.shape[-1]]
        mask_ = np.squeeze(im_merge_t[..., -1])

        return {'buffers': image, 'labels': mask, 'labels_':mask_} 


class Normalize(object):
    # use this normalize method right before ToTensor()
    
    def __call__(self, sample):
        image, mask, mask_ = sample['buffers'], sample['labels'], sample['labels_']

        for n in range(image.shape[0]):
            image[n] = (image[n] - np.mean(image[n])) / (np.std(image[n])+1e-7)

        return {'buffers': image, 'labels': mask, 'labels_':mask_}
