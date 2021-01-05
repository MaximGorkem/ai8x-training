import os
import time

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import mofa_net

import sys
sys.path.insert(0, "../")

from datasets.cifar100 import cifar100_get_datasets


base_arch_params = {}
base_arch_params['in_ch'] = 3
base_arch_params['out_class'] = 100
base_arch_params['n_units'] = 5
base_arch_params['depth_list'] = [4, 3, 3, 3, 2]
base_arch_params['width_list'] = [[64, 64, 64, 64], [64, 64, 128], [128, 128, 128], [128, 128, 128], [128, 128]]
base_arch_params['kernel_list'] = [[3, 3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3]]
base_arch_params['bias_list'] = [[True, True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True]]
base_arch_params['bn'] = True
mofa_base_arch = mofa_net.MOFA_Net_Arch(base_arch_params)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

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

mofa = mofa_net.MOFAnet(mofa_base_arch.get_param_dict(), verbose=False)
mofa = mofa.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mofa.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

best_val_accuracy = 0
max_epochs = 200
for epoch in range(max_epochs):
    t0 = time.time()
    mofa.train()
    for batch, labels in trainset:
        batch, labels = batch.to(device), labels.to(device)
        
        y_pred = mofa(batch)
        loss = criterion(y_pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    print(f'Epoch {epoch+1}')
    print(f'\tTraining loss:{loss.item()}\tLearning Rate:{scheduler.get_last_lr()}')
    t1 = time.time()
    print(f'\tTraining time:{t1-t0:.2f} s - {(t1-t0)/60:.2f} mins ')
    # Validation
    correct = 0
    total = 0
    mofa.eval()
    with torch.no_grad():
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
            os.remove(f'mofa_models/clamp_mofa_acc{100*best_val_accuracy:.0f}%.pth.tar')
        torch.save(mofa, f'mofa_models/clamp_mofa_acc{100*val_accuracy:.0f}%.pth.tar')
        best_val_accuracy = val_accuracy
    print('\tAccuracy of the mofa on the test images: %d %%' % (100 * correct / total))