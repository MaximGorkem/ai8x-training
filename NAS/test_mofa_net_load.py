import os
import time
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import mofa_net


def fuse_bn(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.weight
    gamma = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_bn_mofa(mofa_model):
    param_dict = copy.deepcopy(mofa_model.get_arch())
    param_dict['bn'] = False
    fused_model = mofa_net.MOFAnet(param_dict)
    with torch.no_grad():
        fused_model.classifier.weight.copy_(mofa_model.classifier.weight)
        fused_model.classifier.bias.copy_(mofa_model.classifier.bias)
    for u_ind, unit in enumerate(mofa_model.units):
        for l_ind, layer in enumerate(unit.layers):
            fused_conv = fuse_bn(layer.conv2d, layer.batchnorm)
            fused_conv = fused_conv.to(device)
            with torch.no_grad():
                fused_model.units[u_ind].layers[l_ind].conv2d.weight.copy_(fused_conv.weight)
                fused_model.units[u_ind].layers[l_ind].conv2d.bias.copy_(fused_conv.bias)
    return fused_model

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = torch.load('mofa_models/noclamp_mofa_acc59%.pth.tar', map_location=device)
print(model.get_arch())
fused_model = fuse_bn_mofa(model)
print(fused_model.get_arch())