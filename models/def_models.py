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
from torch.nn.utils import spectral_norm
import torch
from torchvision.models import vgg19
import math
import sys
sys.path.append("../")
from utils.utils import power_iteration
from models.model_blocks import _init_weights
#from model_blocks import conv_layer


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(weights='IMAGENET1K_V1')
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:37])

    def forward(self, img):
        return self.vgg19_54(img)

class VGG19_54(nn.Module):
    def __init__(self, arch_type:str='VGG16', BN:bool=True):
        super(VGG19_54, self).__init__()
        self.arch_type=arch_type
        self.BN = BN
        self.device = torch.device("cuda:0")
        self.in_channel = 3
        Model_list = self.get_arch(arch_type)
        self.conv_layers = self.create_conv_layers(Model_list)
        self.apply(_init_weights)
    def forward(self,x):
        x = self.conv_layers(x)
        return x
    def get_arch(self,name:str)->list:
        if name=='VGG16':
            return [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M']
        else: # else VGG19_54
            return [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
    def create_conv_layers(self, name):
        layers = []
        in_channel = self.in_channel
        for x in name:
            if type(x) == int:
                out_channel = x
                if self.BN:
                    layers+=[Norm(nn.Conv2d(in_channels=in_channel,
                        out_channels = out_channel,
                        kernel_size=3,stride=1,padding=1)),
                        nn.BatchNorm2d(x),
                        nn.ReLU()
                        ]
                else:
                    layers+=[Norm(nn.Conv2d(in_channels=in_channel,
                        out_channels = out_channel,
                        kernel_size=3,stride=1,padding=1)),
                        nn.ReLU()
                        ]
                in_channel = x
            elif x=='M':
                layers+=[nn.MaxPool2d(kernel_size=2,stride=2)]
        return nn.Sequential(*layers)
    def def_loss(self):
        self.loss_FE_fn = nn.L1Loss().to(self.device)
    def def_optimizer(self):
        FE_optim_params = []
        for k,v in self.conv_layers.named_parameters():
            if v.requries_grad:
                FE_optim_params.append(v)
        self.FE_optimizer = Adam(FE_optim_params, lr=self.train_config['lr'], weight_decay=0)

    def feed_data(self, HR, SR):
        # modify this function so that `datas` are taken from output of Gnet  
        self.HR = HR
        self.SR = SR
    def FE_forward(self):
        self.FE_HR = self.conv_layers(self.HR)
        self.FE_SR = self.conv_layers(self.SR)
    def optimize_parameters(self):
        self.FE_optimizer.zero_grad()
        self.FE_forward()
        FE_loss = self.loss_FE_fn(self.SR, self.HR)
        FE_loss.backwards()
        self.FE_optimizer.step()

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
    #property
    def u(self):
        return [getattr(self,'u%i'%i) for i in range(self.num_svs)]

    #property
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
class Conv_SN(nn.Conv2d, SN):
    '''
    This is a warp-up of Conv2d, with singular value buffer registering
    [ERROR] not working, returing a method or class obj?
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding = 0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_iters=1, eps=1e-12, layer_type:str='common'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.conv = nn.Conv2d
        self.conv.__init__(self, in_channels, out_channels, kernel_size, stride,
                     padding, dilation, groups, bias)
        self.conv_method = self.make_conv_layer(layer_type)
        SN.__init__(self, num_svs, num_iters, out_channels, eps=eps) # sv buffer is registered here

    def forward(self,x):
        #return F.conv2d(x, self.W_(), self.bias ,self.stride, self.padding, self.dilation, self.groups)
        return self.conv_method

    def make_conv_layer(self, layer_type):
        if layer_type == 'common':
            return self.conv(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                             self.dilation, self.groups, )
        elif layer_type == 'norm':
            return Norm(self.conv(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                             self.dilation, self.groups, ))
    

# TODO:
# write function scope training and updating for Gnet
class Generator(nn.Module): # interpolation scheme happens
    # A Generator inheritating SN for saving singular value of conv2d
    def __init__(self, conv_method = Conv_SN, channels=3, filters=64, num_res_blocks=32, num_upsample=1):
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
        
        self.apply(_init_weights)
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
    def __init__(self, input_shape:tuple = (3,64,64), conv_method = Conv_SN):
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
        self.apply(_init_weights)

    def forward(self, img):
        return self.model(img)

"""[ERROR]: using Norm(norm(nn.Conv2d)) causes weights not transferred to GPU"""
class Discriminator_Unet(nn.Module):
    """ a Unet with SpectraNorm but no skip connection"""
    def __init__(self, input_shape:tuple=(3,64,64), num_feature=64,skip_connection=True, n_down_layers:int=3, n_up_layers:int=3):
        super(Discriminator_Unet, self).__init__()
        self.norm = spectral_norm
        self.in_channel = input_shape[0]
        self.input_shape = input_shape
        self.output_shape = (1, int(input_shape[1]/2**2), int(input_shape[2]/2**2))
        layers_down = nn.ModuleList([])
        layers_up = nn.ModuleList([])
        self.conv_in = Norm(nn.Conv2d(self.in_channel, num_feature, 3, 1, 1))
        self.num_features = num_feature
        
        # down module
        #layers_down.extend([Norm(nn.Conv2d(num_feature*2**i,
        #   num_feature*2*2**i, 4, 2, 1, bias=False)),nn.LeakyReLU(0.2, inplace=True)] for i in range(n_down_layers))
        [layers_down.extend([Norm(nn.Conv2d(num_feature*2**i,
            num_feature*2*2**i, 4, 2, 1, bias=False)),nn.LeakyReLU(0.2, inplace=True)]) for i in range(n_down_layers)]
        #layers_down = self.make_down_layers(n_down_layers)
        self.down = nn.Sequential(*layers_down)

        # up module
        [layers_up.extend([Norm(nn.Conv2d(num_feature*2*2**i,
                                                    num_feature*2**i, 3, 1, 1, bias=False)),nn.LeakyReLU(0.2, inplace=True)]) for i in range(n_up_layers-1,-1,-1)]
        #[layers_up.extend([Norm(self.norm(nn.Conv2d(num_feature*2*2**i,
         #                                           num_feature*2**i, 3, 1, 1, bias=False))),nn.LeakyReLU(0.2, inplace=True)]) for i in range(n_up_layers-1,-1,-1)]
        #layers_up = self.make_up_layers(n_up_layers)
        self.up = nn.Sequential(*layers_up)

        self.conv_out = Norm(nn.Conv2d(num_feature, 1, 3, 1, 1))
        self.apply(_init_weights)
    def forward(self,x):
        x = self.conv_in(x)
        x_low = self.down(x)
        x_high = F.interpolate(x_low, scale_factor=2, mode='bilinear', align_corners = False)
        x_high = self.up(x_high)
        return self.conv_out(x_high)

    def make_up_layers(self, n_layers):
        layers_up = []
        return [layers_up.extend([self.norm(nn.Conv2d(self.num_features*2*2**i,
            self.num_features*2**i, 3, 1, 1, bias=False)),nn.LeakyReLU(0.2, inplace=True)]) for i in range(n_layers-1,-1,-1)]

    def make_down_layers(self, n_layers):
        layers_down = []
        return [layers_down.extend([self.norm(nn.Conv2d(self.num_features*2**i,
            self.num_features*2*2**i, 4, 2, 1, bias=False)),nn.LeakyReLU(0.2, inplace=True)]) for i in range(n_layers)]
        self.down = nn.Sequential(*layers_down)

class Discriminator_SN_SC(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch=3, num_feat=64, input_shape = (3,64,64),skip_connection=True):
        super(Discriminator_SN_SC, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        self.input_shape = input_shape
        self.output_shape = (1, input_shape[1], input_shape[2])
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        self.apply(_init_weights)
    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
