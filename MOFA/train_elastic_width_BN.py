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
from mofa import *

from datasets.cifar100 import cifar100_get_datasets

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")



# Elastic Width Training

n_units = 5

model = torch.load('mofa_models/arch_1/EW_over4_clamp_mofa_acc63%_ep20250_BN.pth.tar')

# for unit in model.units:    
#     for layer in unit.layers:
#         layer.in_order = torch.arange(layer.conv2d.weight.shape[1])
#         layer.out_order = torch.arange(layer.conv2d.weight.shape[0])


mofa = copy.deepcopy(model)
model = model.to(device)
mofa = mofa.to(device)
model.train()

kd_ratio = 0.5

class Args():
    def __init__(self):
        super(Args, self).__init__()
        self.truncate_testset = False
        self.act_mode_8bit = False
        
args = Args()
train_dataset, test_dataset = cifar100_get_datasets(('../data', args))

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
start_epoch = 20251
max_epochs = 200000
best_val_accuracy = 0
optimizer = torch.optim.SGD(mofa.parameters(), lr=1e-3, momentum=0.9)

for epoch in range(start_epoch, max_epochs):
    t0 = time.time()
    mofa.train()
    for batch, labels in trainset:
        batch, labels = batch.to(device), labels.to(device)
        
        if epoch < 3000:
            mofa = sample_subnet_width(mofa, stage_no=1)
        else:
            mofa = sample_subnet_width(mofa, stage_no=2)

        mofa = sort_channels(mofa)
      
        y_pred = mofa(batch)
        
        if kd_ratio > 0:
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
        
        mofa = update_mofa_from_subnet_width(mofa)
        
    print(f'Epoch {epoch+1}')
    print(f'\tTraining loss:{loss.item()}')
    print(f'\tFirst ktm: {mofa.units[0].layers[0].ktm[4].item()}')
    print(f'\tLast ktm: {mofa.units[-1].layers[-1].ktm[4].item()}')\
    
    t1 = time.time()
    print(f'\tTraining time:{t1-t0:.2f} s - {(t1-t0)/60:.2f} mins ')
    if epoch % 250 == 0:
        mofa = sort_channels(mofa)
        val_accuracy = check_accuracy(mofa, trainset, valset, device=device, bn=True)
        torch.save(mofa, f'mofa_models/arch_1/EW_over4_clamp_mofa_acc{100*val_accuracy:.0f}%_ep{epoch}_BN.pth.tar')
