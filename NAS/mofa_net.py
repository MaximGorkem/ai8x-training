import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FusedConv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, bn=True, verbose=False):
        super(FusedConv2dReLU, self).__init__()
        self.verbose = verbose
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
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                bias=bias)
        
        self.bn = bn
        if self.bn:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.clamp = Clamp(min_val=-1, max_val=1)

    def forward(self, x):
        #print(self.out_channels, self.in_channels, self.conv2d.weight.shape, self.conv2d.bias.shape)
        weight = self.conv2d.weight[:self.out_channels, :self.in_channels, :, :]
        #bias = self.conv2d.bias[:self.out_channels, :self.in_channels, :, :]
        bias = self.conv2d.bias[:self.out_channels]
        if self.kernel_size == 1:
            flattened_weight = weight.view(weight.size(0), weight.size(1), -1, 9)
            #weight = flattened_weight.to(device) @ self.ktm.to(device)
            weight = flattened_weight @ self.ktm
                    
        if self.verbose:
            print(f'Data Shape: {x.shape}\tWeight Shape: {weight.shape}')
        x = self.func(x, weight, bias, self.conv2d.stride, self.pad)
        if self.bn:
            x = self.batchnorm(x)
        x = self.activation(x)
#         x = self.clamp(x)
        return x


class Unit(nn.Module):
    def __init__(self, depth, kernel_list, width_list, init_width, bias_list, bn=True, verbose=False):
        super(Unit, self).__init__()
        self.verbose = verbose
        self.depth = depth
        self.kernel_list = kernel_list
        self.width_list = width_list
        self.bias_list = bias_list
        self.bn = bn
        self._width_list = [init_width] + width_list
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(FusedConv2dReLU(self._width_list[i],
                                               self._width_list[i+1],
                                               self.kernel_list[i],
                                               self.bias_list[i],
                                               self.bn,
                                               self.verbose))
    def forward(self, x):
        for i in range(self.depth):
            x = self.layers[i](x)
        return x


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

    @classmethod
    def create_largest(cls, in_ch=3, out_class=100, n_units=5, max_depth=3, max_width=256, max_kernel=3, bias=True, bn=False):
        param_dict = {}
        param_dict['in_ch'] = in_ch
        param_dict['out_class'] = out_class
        param_dict['n_units'] = n_units
        param_dict['depth_list'] = [max_depth for _ in range(n_units)]
        param_dict['width_list'] = [[max_width]*max_depth for _ in range(n_units)]
        param_dict['kernel_list'] = [[max_kernel]*max_depth for _ in range(n_units)]
        param_dict['bias_list'] = [[bias]*max_depth for _ in range(n_units)]
        param_dict['bn'] = bn
        return cls(param_dict)

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

    def sample_kernel(self):
        param_dict = copy.deepcopy(self.get_param_dict())
        for u_ind in range(param_dict['n_units']):
            for l_ind in range(param_dict['depth_list'][u_ind]):
                possible_kernel_list = [x for x in range(1, param_dict['kernel_list'][u_ind][l_ind]+1, 2)]
                param_dict['kernel_list'][u_ind][l_ind] = random.choice(possible_kernel_list)

        return MOFA_Net_Arch(param_dict)

    def sample_depth(self, sample_kernel=True):
        if sample_kernel:
            param_dict = self.sample_kernel().get_param_dict()
        else:
            param_dict = copy.deepcopy(self.get_param_dict())
        
        for u_ind in range(param_dict['n_units']):
            min_depth = 1
            max_depth = param_dict['depth_list'][u_ind]
            depth = random.randint(min_depth, max_depth)
            param_dict['depth_list'][u_ind] = depth
            param_dict['kernel_list'][u_ind] = param_dict['kernel_list'][u_ind][:depth]
            param_dict['width_list'][u_ind] = param_dict['width_list'][u_ind][:depth]
            param_dict['bias_list'][u_ind] = param_dict['bias_list'][u_ind][:depth]
        
        return MOFA_Net_Arch(param_dict)

    def sample_width(self, sample_kernel_depth=True):
        last_layer_width = self.width_list[-1][-1]

        if sample_kernel_depth:
            param_dict = self.sample_depth(sample_kernel=True).get_param_dict()
        else:
            param_dict = copy.deepcopy(self.get_param_dict())
        
        for u_ind in range(param_dict['n_units']):
            for l_ind in range(param_dict['depth_list'][u_ind]):
                if u_ind == (param_dict['n_units'] - 1) and l_ind == (param_dict['depth_list'][u_ind] - 1):
                    width = last_layer_width
                else:
                    max_width = param_dict['width_list'][u_ind][l_ind]
                    possible_width_list = [int(.5*max_width), int(.75*max_width), max_width]
                    width = random.choice(possible_width_list)
                param_dict['width_list'][u_ind][l_ind] = width
    
        return MOFA_Net_Arch(param_dict)

    def mutate(self, prob_mutation):
        param_dict = copy.deepcopy(self.get_param_dict())

        depth_list = param_dict['depth_list']
        width_list = param_dict['width_list']
        kernel_list = param_dict['kernel_list']
        bias_list = param_dict['bias_list']

        #mutate model depth
        for unit_idx in range(param_dict['n_units']):
            if random.random() < prob_mutation:
                depth = random.choice([1, 2, 3])
                if depth <= depth_list[unit_idx]:
                    width_list[unit_idx] = width_list[unit_idx][:depth]
                    kernel_list[unit_idx] = kernel_list[unit_idx][:depth]
                    bias_list[unit_idx] = bias_list[unit_idx][:depth]
                else:
                    for _ in range(depth - depth_list[unit_idx]):
                        kernel_list[unit_idx].append(random.choice([1, 3]))
                        width_list[unit_idx].append(random.choice([16, 32, 64, 128, 256]))
                        bias_list[unit_idx].append(True)
                depth_list[unit_idx] = depth

        #mutate layer parameters
        for unit_idx in range(len(width_list)):
            for layer_idx in range(len(width_list[unit_idx])):
                if random.random() < prob_mutation:
                    kernel_list[unit_idx][layer_idx] = random.choice([1, 3])
                    width_list[unit_idx][layer_idx] = random.choice([16, 32, 64, 128, 256])

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


class MOFAnet(nn.Module):
    # Maxim OFA Net
    def __init__(self, param_dict, verbose=False):
        super(MOFAnet, self).__init__()
        self.base_arch = copy.deepcopy(param_dict)
        self.verbose = verbose

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
                                   self.bn,
                                   self.verbose))
            self.last_width = self.width_list[i][-1]
        self.flatten = nn.Flatten()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.classifier = nn.Linear(512, self.out_class) 
    
    def update_arch(self, param_dict):
        self.in_ch = param_dict['in_ch']
        self.out_class = param_dict['out_class']
        self.n_units = param_dict['n_units']
        self.depth_list = copy.deepcopy(param_dict['depth_list'])
        self.width_list = copy.deepcopy(param_dict['width_list'])
        self.kernel_list = copy.deepcopy(param_dict['kernel_list'])
        self.bias_list = copy.deepcopy(param_dict['bias_list'])
        self.bn = param_dict['bn']

        for unit_idx, depth in enumerate(self.depth_list):
            self.units[unit_idx].depth = depth
            for layer_idx in range(depth):
                layer = self.units[unit_idx].layers[layer_idx]
                layer.kernel_size = self.kernel_list[unit_idx][layer_idx]
                if layer.kernel_size == 1:
                    layer.pad = 0
                elif layer.kernel_size == 3:
                    layer.pad = 1
                #TODO: change bias of a layer
                if not (unit_idx == 0 and layer_idx == 0):
                    layer.in_channels = last_out_ch
                if unit_idx == (self.n_units-1) and layer_idx == (self.depth_list[unit_idx] - 1):
                    self.width_list[unit_idx][layer_idx] = self.units[-1].layers[-1].conv2d.out_channels
                    layer.out_channels = self.units[-1].layers[-1].conv2d.out_channels
                else:
                    layer.out_channels = self.width_list[unit_idx][layer_idx]
                    last_out_ch = layer.out_channels

    def reset_arch(self):
        self.update_arch(self.base_arch)

    def get_arch(self):
        return {'in_ch': self.in_ch, 'out_class': self.out_class, 'n_units': self.n_units,
                'depth_list': self.depth_list, 'width_list': self.width_list, 
                'kernel_list': self.kernel_list, 'bias_list': self.bias_list, 'bn': self.bn}

    def forward(self, x):
        for i, unit in enumerate(self.units[:-1]):
            x = unit(x)
            x = self.max_pool(x)
        x = self.units[-1](x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
