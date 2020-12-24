import os
import time
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import mofa_net

import sys
sys.path.insert(0, "../")

from datasets.cifar100 import cifar100_get_datasets


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


def fuse_bn_mofa(mofa):
    param_dict = copy.deepcopy(mofa.get_arch())
    param_dict['bn'] = False
    fused_model = mofa_net.MOFAnet(param_dict)
    with torch.no_grad():
        fused_model.classifier.weight.copy_(mofa.classifier.weight)
        fused_model.classifier.bias.copy_(mofa.classifier.bias)
    for u_ind, unit in enumerate(mofa.units):
        for l_ind, layer in enumerate(unit.layers):
            fused_conv = fuse_bn(layer.conv2d, layer.batchnorm)
            fused_conv = fused_conv.to(device)
            with torch.no_grad():
                fused_model.units[u_ind].layers[l_ind].conv2d.weight.copy_(fused_conv.weight)
                fused_model.units[u_ind].layers[l_ind].conv2d.bias.copy_(fused_conv.bias)
    return fused_model


def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = torch.load('mofa_models/noclamp_mofa_acc59%.pth.tar', map_location=device)
mofa_base = fuse_bn_mofa(model)
mofa_base.to(device)
mofa_base_arch = mofa_net.MOFA_Net_Arch(mofa_base.get_arch())
print(mofa_base_arch)

kd_ratio = 0.5

class Args():
    def __init__(self):
        super(Args, self).__init__()
        self.truncate_testset = False
        self.act_mode_8bit = False
        
args = Args()
train_dataset, test_dataset = cifar100_get_datasets(('../data', args))

trainset = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True, num_workers=0)
valset = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False, num_workers=0)

criterion = torch.nn.CrossEntropyLoss()

best_val_accuracy = 0
max_epochs = 2000

for epoch in range(max_epochs):
    t0 = time.time()
    mofa_base.train()
    for batch, labels in trainset:
        batch, labels = batch.to(device), labels.to(device)

        subnet_kernel_arch = mofa_base_arch.sample_kernel()
        mofa_base.update_arch(subnet_kernel_arch.get_param_dict())

        optimizer = torch.optim.SGD(mofa_base.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
      
        y_pred = mofa_base(batch)
        
        if kd_ratio > 0:
            mofa_base.reset_arch()
            with torch.no_grad():
                soft_logits = mofa_base(batch).detach()
                soft_label = F.softmax(soft_logits, dim=1)
            kd_loss = cross_entropy_loss_with_soft_target(y_pred, soft_label)
            loss = kd_ratio * kd_loss + criterion(y_pred, labels)
        else:
            loss = criterion(y_pred, labels)     
        
        mofa_base.update_arch(subnet_kernel_arch.get_param_dict())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mofa_base.reset_arch()
        
    print(f'Epoch {epoch+1}')
    print(f'\tTraining loss:{loss.item()}')
    t1 = time.time()
    print(f'\tTraining time:{t1-t0:.2f} s - {(t1-t0)/60:.2f} mins ')
    
    # Validation
    correct = 0
    total = 0
    mofa_base.eval()
    with torch.no_grad():
        for data in valset:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = mofa_base(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = correct / total
    if val_accuracy > best_val_accuracy:
        if epoch != 0:
            os.remove(f'mofa_models/mofa_ks_acc{100*best_val_accuracy:.0f}%.pth.tar')
        torch.save(mofa_base, f'mofa_models/mofa_ks_acc{100*val_accuracy:.0f}%.pth.tar')
        best_val_accuracy = val_accuracy
    print('\tAccuracy of the mofa on the test images: %d %%' % (100 * correct / total))
    print(f'\tFirst ktm: {mofa_base.units[0].layers[0].ktm[4].item()}')
    print(f'\tLast ktm: {mofa_base.units[4].layers[2].ktm[4].item()}')