import copy
import random
import os
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append('../')

from datasets.cifar100 import cifar100_get_datasets

###

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

### CLASSES ###

class Clamp(nn.Module):
    """
    Post-Activation Clamping Module
    Clamp the output to the given range (typically, [-128, +127])
    """
    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        return x.clamp(min=self.min_val, max=self.max_val)
    
    
class MOFAnet(nn.Module):
    # Maxim OFA Net
    def __init__(self, param_dict):
        super(MOFAnet, self).__init__()
        self.param_dict = param_dict
        self.in_ch = param_dict['in_ch']
        self.out_class = param_dict['out_class']
        self.n_units = param_dict['n_units']
        self.width_list = param_dict['width_list']
        self.kernel_list = param_dict['kernel_list']
        self.bias_list = param_dict['bias_list']
        self.bn = param_dict['bn']
        self.last_width = self.in_ch
        self.units = nn.ModuleList([])
        if 'depth_list' in param_dict:
            self.depth_list = param_dict['depth_list']
        else:
            self.depth_list = []
            for i in range(self.n_units):
                self.depth_list.append(len(self.kernel_list[i]))
        for i in range(self.n_units):
            self.units.append(Unit(self.depth_list[i], 
                                   self.kernel_list[i],
                                   self.width_list[i], 
                                   self.last_width, 
                                   self.bias_list[i],
                                   self.bn))
            self.last_width = self.width_list[i][-1]
        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.classifier = nn.Linear(512, self.out_class) 
    def forward(self, x):
        for i, unit in enumerate(self.units[:-1]):
            x = unit(x)
            x = self.max_pool(x)
        x = self.units[-1](x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    
class Unit(nn.Module):
    def __init__(self, depth, kernel_list, 
                 width_list, init_width, bias_list, bn=True):
        super(Unit, self).__init__()
        self.depth = depth
        self.kernel_list = kernel_list
        self.width_list = width_list
        self.bias_list = bias_list
        self.bn = bn
        self._width_list = [init_width] + width_list
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                FusedConv2dReLU(self._width_list[i],
                                self._width_list[i+1],
                                self.kernel_list[i],
                                self.bias_list[i],
                                self.bn))
    def forward(self, x):
        for i in range(self.depth):
            x = self.layers[i](x)
        return x
    
    
class FusedConv2dReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            bias=True,
            bn=True):
        super(FusedConv2dReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
            
        ktm_core = torch.zeros((9, 1))
        ktm_core[4] = 1
        self.ktm = nn.Parameter(data=ktm_core, requires_grad=True)
        
        if kernel_size == 1:
            self.pad = 0
        elif kernel_size == 3:
            self.pad = 1
        else:
            raise ValueError
        self.func = F.conv2d
        self.conv2d = nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, stride=1,
                                padding=1, bias=bias)
        self.bn = bn
        if self.bn:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.clamp = Clamp(min_val=-1, max_val=1)
    def forward(self, x):        
        weight = self.conv2d.weight[:self.out_channels, :self.in_channels, :, :]
        bias = self.conv2d.bias[:self.out_channels]
        if self.kernel_size == 1:
            flattened_weight = weight.view(weight.size(0), weight.size(1), -1, 9)
            weight = flattened_weight.to(device) @ self.ktm.to(device)
                    
        x = self.func(x, weight, bias, self.conv2d.stride, self.pad)
        if self.bn:
            x = F.batch_norm(x, self.batchnorm.running_mean[:self.out_channels],
                             self.batchnorm.running_var[:self.out_channels],
                             self.batchnorm.weight[:self.out_channels],
                             self.batchnorm.bias[:self.out_channels],
                             self.batchnorm.training,
                             self.batchnorm.momentum,
                             self.batchnorm.eps)

            # x = self.batchnorm(x)
            x = x / 4
        x = self.activation(x)
        x = self.clamp(x)
        return x
    
### FUNCTIONS ###

def make_bn_stats_false(model):
    for u_ind, unit in enumerate(model.units):
        for l_ind, layer in enumerate(unit.layers):
            model.units[u_ind].layers[l_ind].batchnorm.track_running_stats = False

    return model

def make_bn_stats_true(model):
    for u_ind, unit in enumerate(model.units):
        for l_ind, layer in enumerate(unit.layers):
            model.units[u_ind].layers[l_ind].batchnorm.track_running_stats = True

    return model

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
    beta = beta * 0.25
    gamma = gamma * 0.25
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


def fuse_bn_mofa(mofa_net):
    param_dict = copy.deepcopy(mofa_net.param_dict)
    if param_dict['bn'] == False:
        return mofa_net
    else:
        param_dict['bn'] = False
        fused_model = MOFAnet(param_dict)
        with torch.no_grad():
            fused_model.classifier.weight.copy_(mofa_net.classifier.weight)
            fused_model.classifier.bias.copy_(mofa_net.classifier.bias)
        for u_ind, unit in enumerate(mofa_net.units):
            for l_ind, layer in enumerate(unit.layers):
                fused_conv = fuse_bn(layer.conv2d, layer.batchnorm)
                fused_conv = fused_conv.to(device)
                with torch.no_grad():
                    fused_model.units[u_ind].layers[l_ind].conv2d.weight.copy_(fused_conv.weight)
                    fused_model.units[u_ind].layers[l_ind].conv2d.bias.copy_(fused_conv.bias)
                    fused_model.units[u_ind].layers[l_ind].ktm.data.copy_(layer.ktm.data)
        return fused_model

# Elastic Stuff

def sample_subnet_kernel(mofa):
    param_dict = mofa.param_dict
    for u_ind, unit in enumerate(mofa.units):
        for l_ind, layer in enumerate(unit.layers):
            param_dict['kernel_list'][u_ind][l_ind] = random.choice([1, 3])
            layer.kernel_size = param_dict['kernel_list'][u_ind][l_ind]
            if layer.kernel_size == 1:
                layer.pad = 0
    return mofa

def update_mofa_from_subnet_kernel(mofa):
    param_dict = mofa.param_dict
    param_dict['kernel_list'] = []
    for u_ind, unit in enumerate(mofa.units):
        param_dict['kernel_list'].append([])
        for l_ind, layer in enumerate(unit.layers):
            param_dict['kernel_list'][u_ind].append(layer.conv2d.kernel_size[0])
            layer.kernel_size = layer.conv2d.kernel_size
            layer.pad = layer.conv2d.padding
    return mofa

def sample_subnet_depth(mofa, sample_kernel=True):
    if sample_kernel:
        mofa = sample_subnet_kernel(mofa)
    param_dict = mofa.param_dict
    for u_ind, unit in enumerate(mofa.units):
        max_depth = param_dict['depth_list'][u_ind]
        min_depth = 1
        random_depth = random.randint(min_depth, max_depth)
        param_dict['depth_list'][u_ind] = random_depth
        param_dict['kernel_list'][u_ind] = param_dict['kernel_list'][u_ind][:random_depth]
        param_dict['width_list'][u_ind] = param_dict['width_list'][u_ind][:random_depth]
        param_dict['bias_list'][u_ind] = param_dict['bias_list'][u_ind][:random_depth]
        unit.depth = param_dict['depth_list'][u_ind]
    return mofa

def update_mofa_from_subnet_depth(mofa):
    mofa = update_mofa_from_subnet_kernel(mofa)
    param_dict = mofa.param_dict
    param_dict['width_list'] = []
    param_dict['bias_list'] = []
    for u_ind, unit in enumerate(mofa.units):
        max_depth = len(mofa.kernel_list[u_ind])
        param_dict['depth_list'][u_ind] = max_depth
        unit.depth = max_depth
        param_dict['width_list'].append([])
        param_dict['bias_list'].append([])
        for l_ind, layer in enumerate(unit.layers):
            param_dict['width_list'][u_ind].append(layer.conv2d.out_channels)
            param_dict['bias_list'][u_ind].append(layer.conv2d.bias is not None)
    return mofa


def sample_subnet_width(mofa, stage_no=None, possible_width_list=None, sample_kernel_depth=True):
    if sample_kernel_depth:
        mofa = sample_subnet_depth(mofa, sample_kernel=True)
    param_dict = mofa.param_dict
    for u_ind, unit in enumerate(mofa.units):
        for l_ind in range(param_dict['depth_list'][u_ind]):
            layer = mofa.units[u_ind].layers[l_ind]
            if not(u_ind == 0 and l_ind == 0):
                    layer.in_channels = last_out_ch
            if (u_ind == (param_dict['n_units'] - 1)) and (l_ind == (param_dict['depth_list'][u_ind] - 1)):
                param_dict['width_list'][u_ind][l_ind] = mofa.units[-1].layers[-1].conv2d.out_channels
                layer.out_channels = mofa.units[-1].layers[-1].conv2d.out_channels
            else:
                if possible_width_list is not None:
                    pos_width_list = np.array(possible_width_list)
                    valid_inds = pos_width_list <= mofa.units[u_ind].layers[l_ind].conv2d.out_channels
                    pos_width_list = pos_width_list[valid_inds]
                else:
                    if stage_no == 1:
                        pos_width_list = [int(1.0 * layer.conv2d.out_channels), 
                                          int(0.75 * layer.conv2d.out_channels)]
                    elif stage_no == 2:
                        pos_width_list = [int(1.0 * layer.conv2d.out_channels), 
                                          int(0.75 * layer.conv2d.out_channels),
                                          int(0.5 * layer.conv2d.out_channels)]
                    else:
                        print('stage_no must be given!')
                random_width = random.choice(pos_width_list)
                param_dict['width_list'][u_ind][l_ind] = random_width
                layer.out_channels = random_width
                last_out_ch = layer.out_channels
    return mofa   


def update_mofa_from_subnet_width(mofa):
    mofa = update_mofa_from_subnet_depth(mofa)
    param_dict = mofa.param_dict
    for u_ind, unit in enumerate(mofa.units):
        for l_ind, layer in enumerate(unit.layers):
            param_dict['width_list'][u_ind][l_ind] = layer.conv2d.out_channels
            layer.out_channels = layer.conv2d.out_channels
            layer.in_channels = layer.conv2d.in_channels
    return mofa


def sort_channels(mofa):
    with torch.no_grad():
        max_unit_ind = len(mofa.units) - 1
        param_dict = mofa.param_dict
        for u_ind, unit in enumerate(mofa.units):
            max_layer_ind = param_dict['depth_list'][u_ind] - 1
            for l_ind in range(param_dict['depth_list'][u_ind]):
                layer = mofa.units[u_ind].layers[l_ind]
                if not((u_ind == (param_dict['n_units'] - 1)) and (l_ind == (param_dict['depth_list'][u_ind] - 1))):
                    importance = torch.sum(torch.abs(layer.conv2d.weight.data), dim=(1, 2, 3))
                    _, inds = torch.sort(importance, descending=True)
                    layer.conv2d.weight.data = layer.conv2d.weight.data[inds, :, :, :]
                    layer.conv2d.bias.data = layer.conv2d.bias.data[inds]
                    layer.out_order = layer.out_order[inds]
                    if layer.bn:
                        layer.batchnorm.weight.data = layer.batchnorm.weight.data[inds]
                        layer.batchnorm.bias.data = layer.batchnorm.bias.data[inds]
                        layer.batchnorm.running_mean.data = layer.batchnorm.running_mean.data[inds]
                        layer.batchnorm.running_var.data = layer.batchnorm.running_var.data[inds]
                    if l_ind < max_layer_ind:
                        in_order =  mofa.units[u_ind].layers[l_ind+1].in_order
                        reset_ind = torch.argsort(in_order)
                        mofa.units[u_ind].layers[l_ind+1].conv2d.weight.data = \
                        mofa.units[u_ind].layers[l_ind+1].conv2d.weight.data[:, reset_ind, :, :][:, layer.out_order, :, :]
                        mofa.units[u_ind].layers[l_ind+1].in_order = layer.out_order
                    elif l_ind == max_layer_ind:
                        in_order =  mofa.units[u_ind+1].layers[0].in_order
                        reset_ind = torch.argsort(in_order)
                        mofa.units[u_ind+1].layers[0].conv2d.weight.data = \
                        mofa.units[u_ind+1].layers[0].conv2d.weight.data[:, reset_ind, :, :][:, layer.out_order, :, :]
                        mofa.units[u_ind+1].layers[0].in_order = layer.out_order
                else:
                    out_order = layer.out_order
                    reset_ind = torch.argsort(out_order)
                    layer.conv2d.weight.data = layer.conv2d.weight.data[reset_ind, :, :, :]
                    layer.conv2d.bias.data = layer.conv2d.bias.data[reset_ind]
                    layer.out_order = layer.out_order[reset_ind]
                    if layer.bn:
                        layer.batchnorm.weight.data = layer.batchnorm.weight.data[reset_ind]
                        layer.batchnorm.bias.data = layer.batchnorm.bias.data[reset_ind]
                        layer.batchnorm.running_mean.data = layer.batchnorm.running_mean.data[reset_ind]
                        layer.batchnorm.running_var.data = layer.batchnorm.running_var.data[reset_ind]
    return mofa


def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

def see_channel_importances(mofa):
    for u_ind, unit in enumerate(mofa.units):
                for l_ind, layer in enumerate(unit.layers):
                    importance = torch.sum(torch.abs(layer.conv2d.weight.data), dim=(1, 2, 3))
                    print(importance)

def check_accuracy(model, trainset, valset, device='cuda:0', bn=True):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        if bn:
            model.train()
            for data in trainset:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
        model.eval()
        for data in valset:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total)


def update_arch(mofa, param_dict):
    mofa2 = copy.deepcopy(mofa)
    mofa2.in_ch = param_dict['in_ch']
    mofa2.out_class = param_dict['out_class']
    mofa2.n_units = param_dict['n_units']
    mofa2.depth_list = copy.deepcopy(param_dict['depth_list'])
    mofa2.width_list = copy.deepcopy(param_dict['width_list'])
    mofa2.kernel_list = copy.deepcopy(param_dict['kernel_list'])
    mofa2.bias_list = copy.deepcopy(param_dict['bias_list'])
    mofa2.bn = param_dict['bn']
    mofa2.param_dict = param_dict

    for unit_idx, depth in enumerate(mofa2.depth_list):
        mofa2.units[unit_idx].depth = depth
        for layer_idx in range(depth):
            layer = mofa2.units[unit_idx].layers[layer_idx]
            layer.kernel_size = mofa2.kernel_list[unit_idx][layer_idx]
            if layer.kernel_size == 1:
                layer.pad = 0
            elif layer.kernel_size == 3:
                layer.pad = 1
            #TODO: change bias of a layer
            if not (unit_idx == 0 and layer_idx == 0):
                layer.in_channels = last_out_ch
            if unit_idx == (mofa2.n_units-1) and layer_idx == (mofa2.depth_list[unit_idx] - 1):
                mofa2.width_list[unit_idx][layer_idx] = mofa2.units[-1].layers[-1].conv2d.out_channels
                layer.out_channels = mofa2.units[-1].layers[-1].conv2d.out_channels
            else:
                layer.out_channels = mofa2.width_list[unit_idx][layer_idx]
                last_out_ch = layer.out_channels

    mofa2 = sort_channels(mofa2)
    return mofa2

class MOFA_Net_Arch():
    def __init__(self, param_dict):
        self.in_ch = param_dict['in_ch']
        self.out_class = param_dict['out_class']
        self.n_units = param_dict['n_units']
        self.depth_list = param_dict['depth_list']
        self.width_list = param_dict['width_list']
        self.kernel_list = param_dict['kernel_list']
        self.bias_list = param_dict['bias_list']
        self.bn = param_dict['bn']

    def get_param_dict(self):
            return {'in_ch': self.in_ch, 'out_class': self.out_class, 'n_units': self.n_units,
                    'depth_list': self.depth_list, 'width_list': self.width_list, 
                    'kernel_list': self.kernel_list, 'bias_list': self.bias_list, 'bn': self.bn}

    def get_num_weights(self):
        num_params = 0
        ch_prev = self.in_ch
        for unit_idx in range(self.n_units):
            for d, width in enumerate(self.width_list[unit_idx]):
                num_params += (ch_prev * width * (self.kernel_list[unit_idx][d]**2))
                ch_prev = width
        num_params += (1024*self.out_class)
        return num_params

    def mutate(self, prob_mutation, mutate_kernel=True, mutate_depth=True, mutate_width=True):
        param_dict = copy.deepcopy(self.get_param_dict())

        depth_list = param_dict['depth_list']
        width_list = param_dict['width_list']
        kernel_list = param_dict['kernel_list']
        bias_list = param_dict['bias_list']

        #mutate model depth
        if mutate_depth:
            for unit_idx in range(param_dict['n_units']):
                if random.random() < prob_mutation:
                    if unit_idx == 0:
                        min_depth = 2
                        max_depth = 4
                    elif unit_idx == (param_dict['n_units'] - 1):
                        min_depth = 1
                        max_depth = 2
                    else:
                        min_depth = 1
                        max_depth = 3
                    #depth = random.choice([1, 2, 3])
                    depth = random.randint(min_depth, max_depth)
                    if depth <= depth_list[unit_idx]:
                        width_list[unit_idx] = width_list[unit_idx][:depth]
                        kernel_list[unit_idx] = kernel_list[unit_idx][:depth]
                        bias_list[unit_idx] = bias_list[unit_idx][:depth]
                    else:
                        for _ in range(depth - depth_list[unit_idx]):
                            kernel_list[unit_idx].append(random.choice([1, 3]))
                            if (unit_idx == 0) or ( (unit_idx == 1) and len(width_list[unit_idx]) < 3):
                                width_opts = [32, 48, 64]
                            else:
                                width_opts = [64, 96, 128]
                            
                            if mutate_width:
                                width_list[unit_idx].append(random.choice(width_opts))
                            else:
                                width_list[unit_idx].append(width_opts[-1])
                            bias_list[unit_idx].append(True)
                    depth_list[unit_idx] = depth
            width_list[-1][-1] = 128

        #mutate layer parameters
        for unit_idx in range(len(width_list)):
            for layer_idx in range(len(width_list[unit_idx])):
                if random.random() < prob_mutation:
                    if mutate_kernel:
                        kernel_list[unit_idx][layer_idx] = random.choice([1, 3])
                    
                    #width_list[unit_idx][layer_idx] = random.choice([16, 32, 64, 128, 256])
                    if (unit_idx == 0) or ( (unit_idx == 1) and layer_idx < 2):
                        width_opts = [32, 48, 64]
                    else:
                        width_opts = [64, 96, 128]
                    
                    if mutate_width:
                        width_list[unit_idx][layer_idx] = random.choice(width_opts)
                    else:
                        width_list[unit_idx][layer_idx] = width_opts[-1]

        width_list[-1][-1] = 128

        return MOFA_Net_Arch(param_dict)


    @staticmethod
    def crossover(model1, model2):
        assert model1.in_ch == model2.in_ch
        assert model1.out_class == model2.out_class
        assert model1.n_units == model2.n_units
        assert model1.bn == model2.bn

        depth_list = []
        width_list = []
        kernel_list = []
        bias_list = []

        #crossover model depths
        for unit_idx in range(model1.n_units):
            depth_list.append(random.choice([model1.depth_list[unit_idx], model2.depth_list[unit_idx]]))

        #crossover layers
        for unit_idx, depth in enumerate(depth_list):
            width_list.append([])
            kernel_list.append([])
            bias_list.append([])
            for d in range(depth):
                if d >= model1.depth_list[unit_idx]:
                    width_list[unit_idx].append(model2.width_list[unit_idx][d])
                    kernel_list[unit_idx].append(model2.kernel_list[unit_idx][d])
                    bias_list[unit_idx].append(model2.bias_list[unit_idx][d])
                elif d >= model2.depth_list[unit_idx]:
                    width_list[unit_idx].append(model1.width_list[unit_idx][d])
                    kernel_list[unit_idx].append(model1.kernel_list[unit_idx][d])
                    bias_list[unit_idx].append(model1.bias_list[unit_idx][d])
                else:
                    width_list[unit_idx].append(random.choice([model1.width_list[unit_idx][d], model2.width_list[unit_idx][d]]))
                    kernel_list[unit_idx].append(random.choice([model1.kernel_list[unit_idx][d], model2.kernel_list[unit_idx][d]]))
                    bias_list[unit_idx].append(random.choice([model1.bias_list[unit_idx][d], model2.bias_list[unit_idx][d]]))
        
        param_dict = {'in_ch': model1.in_ch, 'out_class': model1.out_class, 'n_units': model1.n_units,
                      'depth_list': depth_list, 'width_list': width_list, 'kernel_list': kernel_list, 'bias_list': bias_list,
                      'bn': model1.bn}
        
        return MOFA_Net_Arch(param_dict)
