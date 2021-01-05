import os
import time
import copy

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import mofa_net

import sys
sys.path.insert(0, "../")

from datasets.cifar100 import cifar100_get_datasets


def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#mofa_base = torch.load('mofa_models/Clamp_Trial1/mofa_ks_acc67%.pth.tar', map_location=device)
mofa_base = torch.load('mofa_models/mofa_ds_lev1_ep2000_acc60%.pth.tar', map_location=device)
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

level = 2
best_val_accuracy = 0
max_epochs = 20000
init_epochs = 0

optimizer = torch.optim.SGD(mofa_base.parameters(), lr=1e-3, momentum=0.9)
#max_optim_steps = 50
#optim_step = 0
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_optim_steps, eta_min=1e-5)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 15000], gamma=0.5)

for epoch in range(init_epochs, init_epochs+max_epochs):
    t0 = time.time()
    mofa_base.train()
    for batch, labels in trainset:
        batch, labels = batch.to(device), labels.to(device)

        subnet_depth_arch = mofa_base_arch.sample_depth(level=level)
        mofa_base.update_arch(subnet_depth_arch.get_param_dict())

        y_pred = mofa_base(batch)

        if kd_ratio > 0:
            mofa_base.reset_arch()
            with torch.no_grad():
                soft_logits = mofa_base(batch).detach()
                soft_label = F.softmax(soft_logits, dim=1)
            kd_loss = cross_entropy_loss_with_soft_target(y_pred, soft_label)
            loss = kd_ratio * kd_loss + criterion(y_pred, labels)
            mofa_base.update_arch(subnet_depth_arch.get_param_dict())
        else:
            loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mofa_base.reset_arch()

    scheduler.step()
    # optim_step += 1
    # if optim_step == max_optim_steps:
    #     optimizer = torch.optim.SGD(mofa_base.parameters(), lr=1e-4, momentum=0.9)
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_optim_steps)
    #     optim_step = 0

    print(f'Epoch {epoch + 1}')
    print(f'\tTraining loss:{loss.item()}\tLearning Rate:{scheduler.get_last_lr()}')
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
        if epoch != init_epochs:
            os.remove(f'mofa_models/mofa_ds_lev{level}_acc{100*best_val_accuracy:.0f}%.pth.tar')
        torch.save(mofa_base, f'mofa_models/mofa_ds_lev{level}_acc{100*val_accuracy:.0f}%.pth.tar')
        best_val_accuracy = val_accuracy
    print('\tAccuracy of the mofa on the test images: %d %%' % (100 * correct / total))

    if ((epoch+1) % 500) == 0:
        torch.save(mofa_base, f'mofa_models/mofa_ds_lev{level}_ep{epoch+1}_acc{100*val_accuracy:.0f}%.pth.tar')


torch.save(mofa_base, f'mofa_models/mofa_ds_lev{level}_ep{epoch+1}_acc{100*val_accuracy:.0f}%.pth.tar')