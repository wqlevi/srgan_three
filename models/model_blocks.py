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

# ----- DWT blocks -----#
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        return dwt_init(x)
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        return iwt_init(x)
    
class DWT_transform(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels*3, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        dwt_low_frequency,dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency,dwt_high_frequency


# ----- IWT blocks -----#
def iwt_init(x):
    # completely reverse dwt_init function
    # x : all LL, LH, HL, HH cat along channel
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = (
            in_batch,
            int(in_channel / (r**2)),
            r*in_height,
            r*in_width
            )
    x1 = x[:, 0:out_channel, :, :] /2
    x2 = x[:, out_channel:out_channel *2, :, :] /2
    x3 = x[:, out_channel *2:out_channel *3, :, :] /2
    x4 = x[:, out_channel *3:out_channel *4, :, :] /2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h

class IWT_transform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.iwt = IWT()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

