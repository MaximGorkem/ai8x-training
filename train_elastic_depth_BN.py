import copy
import random
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from datasets.cifar100 import cifar100_get_datasets

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

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
            x = self.batchnorm(x)
            x = x / 4
        x = self.activation(x)
        x = self.clamp(x)
        return x

# Functions

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
    beta = beta * 0.25
    gamma = bn.bias
    gamma = gamma * 0.25
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


def sample_subnet_width(mofa, possible_width_list, sample_kernel_depth=True):
    if sample_kernel_depth:
        mofa = sample_subnet_depth(mofa, sample_kernel=True)
    param_dict = mofa.param_dict
    for u_ind, unit in enumerate(mofa.units):
        for l_ind in range(param_dict['depth_list'][u_ind]):
            layer = mofa.units[u_ind].layers[l_ind]
            if not(u_ind == 0 and l_ind == 0):
                    layer.in_channels = last_out_ch
            if u_ind == (param_dict['n_units'] - 1) and l_ind == (param_dict['depth_list'][u_ind] - 1):
                param_dict['width_list'][u_ind][l_ind] = mofa.units[-1].layers[-1].conv2d.out_channels
                layer.out_channels = mofa.units[-1].layers[-1].conv2d.out_channels
            else:
                possible_width_list = np.array(possible_width_list)
                valid_inds = possible_width_list <= mofa.units[u_ind].layers[l_ind].conv2d.out_channels
                possible_width_list = possible_width_list[valid_inds]
                random_width = random.choice(possible_width_list)
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
    for ind in range((mofa.n_units*mofa.param_dict['depth_list'][0])-1):
        u_ind = ind // n_layers
        l_ind = ind % n_layers
        layer = mofa.units[u_ind].layers[l_ind]
        
        importance = torch.sum(torch.abs(layer.conv2d.weight.data), dim=(1, 2, 3))
        _, inds = torch.sort(importance, descending=True)
        layer.conv2d.weight.data = layer.conv2d.weight.data[inds, :, :, :]
        layer.conv2d.bias.data = layer.conv2d.bias.data[inds]
        
        ind_new = ind + 1
        u_ind = ind_new // n_layers
        l_ind = ind_new % n_layers
        mofa.units[u_ind].layers[l_ind].conv2d.weight.data = mofa.units[u_ind].layers[l_ind].conv2d.weight.data[:, inds, :, :]
    return mofa

def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

def see_channel_importances(mofa):
    for u_ind, unit in enumerate(mofa.units):
                for l_ind, layer in enumerate(unit.layers):
                    importance = torch.sum(torch.abs(layer.conv2d.weight.data), dim=(1, 2, 3))
                    print(importance)

# Elastic Depth Training

n_units = 5

model = torch.load('mofa_models/arch_1/Best_EK_over4_clamp_mofa_acc63%_BN.pth.tar')

mofa = copy.deepcopy(model)
model = model.to(device)
mofa = mofa.to(device)

kd_ratio = 0.5

class Args():
    def __init__(self):
        super(Args, self).__init__()
        self.truncate_testset = False
        self.act_mode_8bit = False
        
args = Args()
train_dataset, test_dataset = cifar100_get_datasets(('data', args))

trainset = DataLoader(dataset=train_dataset,
                      batch_size=100,
                      shuffle=True,
                      num_workers=0)

valset = DataLoader(dataset=test_dataset,
                      batch_size=1000,
                      shuffle=False,
                      num_workers=0)

criterion = torch.nn.CrossEntropyLoss()

best_val_accuracy = 0
max_epochs = 10001
for epoch in range(max_epochs):
    t0 = time.time()
    mofa.train()
    for batch, labels in trainset:
        batch, labels = batch.to(device), labels.to(device)
        
        mofa = make_bn_stats_false(mofa)
        subnet = sample_subnet_depth(mofa)
        optimizer = torch.optim.SGD(subnet.parameters(), lr=1e-3, momentum=0.9)
      
        y_pred = subnet(batch)
        
        if kd_ratio > 0:
            model.train()
            with torch.no_grad():
                soft_logits = model(batch).detach()
                soft_label = F.softmax(soft_logits, dim=1)
            kd_loss = cross_entropy_loss_with_soft_target(y_pred, soft_label)
            loss = kd_ratio * kd_loss + criterion(y_pred, labels)
        else:
            loss = criterion(y_pred, labels)     
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        mofa = update_mofa_from_subnet_depth(mofa)
        
    print(f'Epoch {epoch+1}')
    print(f'\tTraining loss:{loss.item()}')
    t1 = time.time()
    print(f'\tTraining time:{t1-t0:.2f} s - {(t1-t0)/60:.2f} mins ')
    
    # Validation
    correct = 0
    total = 0
    mofa.eval()
    with torch.no_grad():
        mofa = make_bn_stats_true(mofa)
        mofa.train()
        for data in valset:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = mofa(images)
        mofa.eval()
        for data in valset:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = mofa(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    if val_accuracy > best_val_accuracy:
        if epoch != 0:
            os.remove(f'mofa_models/arch_1/Best_ED_over4_clamp_mofa_acc{100*best_val_accuracy:.0f}%_BN.pth.tar')
        torch.save(mofa, f'mofa_models/arch_1/Best_ED_over4_clamp_mofa_acc{100*val_accuracy:.0f}%_BN.pth.tar')
        best_val_accuracy = val_accuracy
    print('\tAccuracy of the mofa on the test images: %d %%' % (
        100 * correct / total))
    print(f'\tFirst ktm: {mofa.units[0].layers[0].ktm[4].item()}')
    print(f'\tLast ktm: {mofa.units[-1].layers[-1].ktm[4].item()}')
    if epoch % 500 == 0:
        torch.save(mofa, f'mofa_models/arch_1/ED_over4_clamp_mofa_acc{100*val_accuracy:.0f}%_ep{epoch}_BN.pth.tar')
