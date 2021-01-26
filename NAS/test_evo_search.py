import torch
from torch.utils.data import DataLoader

import mofa_net
import network_search

import sys
sys.path.insert(0, "../")

from datasets.cifar100 import cifar100_get_datasets


class Args():
    def __init__(self):
        super(Args, self).__init__()
        self.truncate_testset = False
        self.act_mode_8bit = False

constraint = {'max_num_weights': 4e5}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#mofa_base = mofa_net.MOFA_Net_Arch.create_largest()
mofa_base = torch.load('mofa_models/mofa_ds_lev2_acc59%.pth.tar', map_location=device)
#mofa_base_arch = mofa_net.MOFA_Net_Arch(mofa_base.get_arch())

args = Args()
_, test_dataset = cifar100_get_datasets(('../data', args))
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False, num_workers=0)

evo_search = network_search.EvolutionSearch(population_size=50, num_iter=10)
evo_search.set_model(mofa_base)
best_arch, best_acc = evo_search.run(constraint, test_loader, device)