#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 01:28:09 2021

- [x] resized NN generator
- [x] SV registered both to Gnet and Dnet
- [ ] Not recording SV correctly
@author: qiwang
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
import sys
sys.path.append("../")
from utils.utils import power_iteration


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=False)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:37])

    def forward(self, img):
        return self.vgg19_54(img)

class VGG19_54(nn.Module):
    def __init__(self):
        super(VGG19_54, self).__init__()
        Model_list = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M']
        self.in_channel = 3
        self.conv_layers = self.create_conv_layers(Model_list)
    def forward(self,x):
        x = self.conv_layers(x)
        return x
    def create_conv_layers(self, name):
        layers = []
        in_channel = self.in_channel
        for x in name:
            if type(x) == int:
                out_channel = x
                layers+=[Norm(nn.Conv2d(in_channels=in_channel,
                        out_channels = out_channel,
                        kernel_size=3,stride=1,padding=1)),
                        nn.BatchNorm2d(x),
                        nn.ReLU()
                        ]
                in_channel = x
            elif x=='M':
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
        return nn.Sequential(*layers)

class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [Norm(nn.Conv2d(in_features, filters, 3, 1, 1, bias=True))]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
    
    

class SN(object):
    '''
    svs : singular values
    sv : also singular values; updated per model forward
    '''
    def __init__(self, num_svs, num_iters,num_outputs,transpose=False,eps=1e-12):
        self.num_iters = num_iters
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps
        for i in range(self.num_svs):
            self.register_buffer('u%d'%i, torch.randn(1,num_outputs))
            self.register_buffer('sv%d'%i, torch.ones(1))
    @property
    def u(self):
        return [getattr(self,'u%i'%i) for i in range(self.num_svs)]

    @property
    def sv(self):
        return [getattr(self,'sv%i'%i) for i in range(self.num_svs)] # create attr
    
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0),-1)
        if self.transpose:
            W_mat = W_mat.t()
        for _ in range(self.num_iters):
            svs,us,vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight/svs[0]
    
class Norm(nn.Module):
    # Updating SV but not for Spectral Norm
    def __init__(self,module,name='weight'):
        super(Norm,self).__init__()
        self.module = module
        self.name = name
        self.module.register_buffer(self.name+'_sv', torch.ones(1))
    def update_v_u(self):
        w = getattr(self.module,self.name)
        sv = getattr(self.module,self.name+'_sv')
        sv[:] = torch.norm(w.detach())
        setattr(self.module, self.name+'_sv', sv)
    def forward(self, *args):
        self.update_v_u()
        return self.module.forward(*args)
# Do SN and record SV
class G_SN(nn.Conv2d, SN):
    '''
    This is a warp-up of Conv2d, with singular value buffer registering
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding = 0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_iters=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                       padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_iters, out_channels, eps=eps) # sv buffer is registered here
    def forward(self,x):
        return F.conv2d(x, self.W_(), self.bias ,self.stride, self.padding, self.dilation, self.groups)
    

    
class Generator(nn.Module): # interpolation scheme happens
    # A Generator inheritating SN for saving singular value of conv2d
    def __init__(self, conv_method = G_SN, channels=3, filters=64, num_res_blocks=32, num_upsample=1):
        super(Generator, self).__init__()
        
        
        self.conv_method = conv_method
        # First layer
        
        #self.conv1 = self.conv_method(channels, filters, kernel_size=3, stride=1, padding=1)
        self.conv1 = Norm(nn.Conv2d(channels, filters, kernel_size=3,stride=1,padding=1))
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        
        self.conv2 = Norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1))
        #self.conv2 = nn.Conv2d(filters, filters, kernel_size=3,stride=1,padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Upsample(scale_factor=2,mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.LeakyReLU(),
                Norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0))
            ]
            
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        
        '''
        self.conv3 = nn.Sequential(
            self.conv_method(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            self.conv_method(filters, channels, kernel_size=3, stride=1, padding=1),
        )
        '''
        self.conv3 = nn.Sequential(
            Norm(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
            Norm(nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1)),
        )
        
        #self.apply(self.init_weights)
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out
    def init_weights(self,m):
        if isinstance(m,nn.Conv2d):
            #torch.nn.init.xavier_uniform(m.weight)
            torch.nn.init.constant_(m.weight,0.0241)
            m.bias.data.fill_(.01)


class Discriminator(nn.Module,):
    def __init__(self, input_shape:tuple, conv_method = G_SN):
        super(Discriminator, self).__init__()
        self.conv_method = conv_method
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(Norm(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(Norm(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(Norm(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1)))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
