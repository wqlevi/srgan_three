#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:41:57 2022

@author: qiwang
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class Gradient(nn.Module):
    """
    Sobel filtering the image to get gradient
    """
    def __init__(self):
        super(Gradient, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()
        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_output = list(map(lambda x : self.comp_grad(x[None,:]), x[:]))
        grad_all = torch.stack(grad_output, dim=0).norm(dim=0) # norm of all color channels
        return grad_all
    def comp_grad(self,x):
        # compute for each color channel
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient



class Lap(nn.Module):
    """
    A Gradient of insensity computation 
    """
    def __init__(self):
        super(Lap,self).__init__()
        kernel_x = [[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)

    def forward(self, x):
        grad_x = list(map(lambda x : F.conv2d(x[None,:], self.weight_x), x[:]))
        gradient = torch.stack(grad_x, dim=0) # not weight summing 2nd grad
        return gradient
