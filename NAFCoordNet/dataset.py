

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
import torch.nn
import skimage.io
import glob
from model import *

import torch
import glob
import skimage.io
from torch.utils import data

class CM2Dataset(data.Dataset):
    def __init__(self, dir_data, transform=None):
        self.dir_data = dir_data
        self.transform = transform
        self.num_stacks = len(glob.glob(self.dir_data + '/demix_*.tif'))  # Number of image stacks
        self.num_stacks = self.num_stacks//3
        self.image_size = 2400  # Assuming the original image is 2400x2400
        
    def __getitem__(self, index):
        # Load the full image stack
        meas_stack = skimage.io.imread(self.dir_data + f'/demix_{index + 1}.tif')
        meas_stack = meas_stack.astype('float32') / meas_stack.max()
        # Load the ground truth image
        gt = skimage.io.imread(self.dir_data + f'/gt_{index + 1}.tif')
        gt = gt.astype('float32') / gt.max()

        # Generate index list for coordinates
        index_list = indexGenerate(0, 0, self.image_size, self.image_size)

        # Return whole image data
        data = {'gt': gt, 'meas': meas, 'index': index_list}
        
        if self.transform:
            data = self.transform(data)
        return data

    def __len__(self):
        return self.num_stacks


class Subset(data.Dataset):
    def __init__(self, dataset, isVal, patch_size=480, stride=240):
        self.dataset = dataset
        self.isVal = isVal
        self.patch_size = 480
        self.stride = 240
        self.image_size = 2400

        # Calculate the number of patches per image
        self.patches_per_row = (self.image_size - self.patch_size) // self.stride + 1
        self.patches_per_image = self.patches_per_row * self.patches_per_row
        self.total_patches = len(self.dataset) * self.patches_per_image

    def __getitem__(self, index):
        # Determine the image stack and the patch within that image
        stack_index = index // self.patches_per_image
        patch_index = index % self.patches_per_image

        # Get the full image data from CM2Dataset
        data = self.dataset[stack_index]

        gt, meas, index_list = data['gt'], data['meas'], data['index']
        
        if self.isVal:
            return self.dataset[index]
        else:
            # Patch-wise operation for training
            row = patch_index // self.patches_per_row
            col = patch_index % self.patches_per_row
            start_y = row * self.stride
            start_x = col * self.stride

            # Extract patches for gt, meas, and index_list
            gt_patch = gt[start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]
            meas_patch = meas[start_y:start_y + self.patch_size, start_x:start_x + self.patch_size]
            index_patch = index_list[start_y:start_y + self.patch_size, start_x:start_x + self.patch_size, :]

            # Return patch-wise data
            return {'gt': gt_patch, 'meas': meas_patch, 'index': index_patch}

    def __len__(self):
        if self.isVal:
            return self.dataset.__len__()
        else:
            return self.total_patches





class ToTensorcm2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        gt, meas,index = data['gt'], data['meas'], data['index']
        return {'gt': torch.from_numpy(gt),
                'meas': torch.from_numpy(meas),
                'index':index
                }

class Noisecm2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        gt, meas, demix = data['gt'], data['meas'], data['demix']
        amin = 7.8109e-5
        amax = 9.6636e-5
        bmin = 1.3836e-8
        bmax = 1.1204e-7
        a = np.random.rand(1) * (amax - amin) + amin  # from calibration
        b = np.random.rand(1) * (bmax - bmin) + bmin  # from calibration
        meas += np.sqrt(a * meas + b) * np.random.randn(meas.shape[0], meas.shape[1])
        data = {'gt': gt, 'meas': meas, 'demix': demix}

        return data



class Crop(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        gt, meas, demix = data['gt'], data['meas'], data['demix']
        tot_len= 2400
        tmp_pad = 900
        meas = F.pad(meas, (tmp_pad, tmp_pad, tmp_pad, tmp_pad), 'constant', 0)

        loc = [(664, 1192), (664, 2089), (660, 2982),
               (1564, 1200), (1557, 2094), (1548, 2988),
               (2460, 1206), (2452, 2102), (2444, 2996)]

        meas = torch.stack([
            meas[x - (tot_len // 2) + tmp_pad:x + (tot_len // 2) + tmp_pad,
            y - (tot_len // 2) + tmp_pad:y + (tot_len // 2) + tmp_pad] for x, y in loc
        ])
        # print(meas.shape,gt.shape,demix.shape)
        data = {'gt': gt, 'meas': meas, 'demix':demix}
        return data
