"""
- [ ] continue to finish post-hoc analysis of FE 
"""
import torch
from models.new_model import FeatureExtractor
from dataset import dataloader
import numpy as np

device = torch.device("cuda:0")
img_size = 64
batch_size = 16
dst = dataloader(root="/big_data/qi1/Celeba/train", hr_shape=img_size)
loader = torch.utils.data.dataloader.DataLoader(dst, batch_size = batch_size, shuffle=True, num_workers=2, drop_last=True)

FE = FeatureExtractor().to(device)
_,im = next(enumerate(loader))
