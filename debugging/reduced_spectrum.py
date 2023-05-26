#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:01:59 2023

@author: qiwang
"""
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor as ptt
from spectrum import *
import matplotlib.pyplot as plt
import numpy as np
def to_gray_tensor(img):
    im = img.convert('L')
    return ptt(im).to(torch.float32)

#path_list = ['../SR_ours.png',
#             '../../SwinIR/testsets/set5/HR/img_001.png',
#             '../../SwinIR/set5test_results/swinir_classical_sr_x2/img_001_SwinIR.png']
path_list = ['../MRI_dataset/GT.png',
             '../MRI_dataset/Linear.png',
             '../MRI_dataset/Ours.png',
             '../MRI_dataset/UNet.png',
             '../MRI_dataset/WGAN.png',
             '../MRI_dataset/ESRGAN_norm.png']
im_ls = list(map(Image.open,path_list))
img_ls = list(map(to_gray_tensor, im_ls))

norm = lambda x : (x-x.mean())/x.std()
img_ls_n = list(map(norm, img_ls))
fft_ls = list(map(batch_fft, img_ls_n))
spectrum_ls = list(map(get_spectrum, fft_ls))

#colour = ['r','g','b']
#names = ['Ours','GT', 'SwinIR']
names = ['GT','Linear','Ours','UNet','WGAN','ESRGAN']
colour = ['k','g','r','purple','b','y']
for i,x in enumerate(spectrum_ls):
    if i==0:
        plt.plot(np.linspace(1,spectrum_ls[0].shape[1],spectrum_ls[0].shape[1]), x.squeeze(),
                 c= colour[i],
                 linewidth = 2,
                 alpha=0.6,
                 linestyle='-',
                 label=names[i])
    else:
        plt.plot(np.linspace(1,spectrum_ls[0].shape[1],spectrum_ls[0].shape[1]), x.squeeze(),
             c= colour[i],
             linewidth = 1.5,
             alpha=0.6,
             linestyle='--',
             label=names[i])
#plt.xlim(0,351)
plt.yscale('log')
plt.xlabel('reduced freq')
plt.ylabel('power intensity')
plt.legend()
plt.show()
