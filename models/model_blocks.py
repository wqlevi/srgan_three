import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_c, out_c, kernel_size, stride=1, dilation=1, bias=True, padding=0):
    return nn.Conv2d(in_c, out_c, kernel_size, stride, padding, dilation, bias)

def _init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias.data, 0.)
