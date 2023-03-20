#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:19:11 2022

@author: qiwang
"""
import torchvision.transforms as transforms
import glob
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
class dataloader(Dataset):
    def __init__(self,root,hr_shape):
        self.root = root
        self.shape = hr_shape
        self.lr_shape = int(hr_shape/2)
        self.lr_transform = transforms.Compose([
            transforms.Resize(self.lr_shape),
            transforms.CenterCrop(self.lr_shape),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))
            ])
        self.hr_transform = transforms.Compose([
            transforms.Resize(self.shape),
            transforms.CenterCrop(self.shape),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))
            ])
        self.files = sorted(glob.glob(self.root+'/*jpg'))
    def __getitem__(self,index):
        img = Image.open(self.files[index % len(self.files)])
        #img = np.array(img)
        im_lr = self.lr_transform(img)
        im_hr = self.hr_transform(img)
        return {"lr":im_lr, "hr":im_hr}
    def __len__(self):
        return len(self.files)

class ffcv_dataloader(Dataset):
    def __init__(self,root,hr_shape):
        self.root = root
        self.shape = hr_shape
        self.lr_shape = int(hr_shape/2)
        self.lr_transform = transforms.Compose([
            transforms.Resize(self.lr_shape),
            transforms.CenterCrop(self.lr_shape),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))
            ])
        self.hr_transform = transforms.Compose([
            transforms.Resize(self.shape),
            transforms.CenterCrop(self.shape),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))
            ])
        self.files = sorted(glob.glob(self.root+'/*jpg'))
    def __getitem__(self,index):
        img = Image.open(self.files[index % len(self.files)])
        #img = np.array(img)
        im_lr = self.lr_transform(img)
        im_hr = self.hr_transform(img)
        return (im_lr.numpy(),im_hr.numpy())
    def __len__(self):
        return len(self.files)
