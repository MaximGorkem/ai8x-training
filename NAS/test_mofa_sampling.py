import torch

import mofa_net

#mofa_base_arch = mofa_net.MOFA_Net_Arch.create_largest()

base_arch_params = {}
base_arch_params['in_ch'] = 3
base_arch_params['out_class'] = 100
base_arch_params['n_units'] = 5
base_arch_params['depth_list'] = [3, 3, 3, 3, 3]
base_arch_params['width_list'] = [[48, 48, 48], [64, 64, 64], [64, 64, 64], [128, 128, 128], [128, 128, 128]]
base_arch_params['kernel_list'] = [[3]*3 for _ in range(5)]
base_arch_params['bias_list'] = [[True]*3 for _ in range(5)]
base_arch_params['bn'] = False
mofa_base_arch = mofa_net.MOFA_Net_Arch(base_arch_params)

print('#######TEST KERNEL#########')
for i in range(10):
    print(mofa_base_arch.sample_kernel().get_param_dict())
    print('###')
print('')

print('#######TEST DEPTH#########')
for i in range(10):
    print(mofa_base_arch.sample_depth(sample_kernel=False).get_param_dict())
    print('###')
print('')
for i in range(10):
    print(mofa_base_arch.sample_depth(sample_kernel=True).get_param_dict())
    print('###')
print('')

print('#######TEST WIDTH#########')
for i in range(10):
    print(mofa_base_arch.sample_width(sample_kernel_depth=False).get_param_dict())
    print('###')
print('')
for i in range(10):
    print(mofa_base_arch.sample_width(sample_kernel_depth=True).get_param_dict())
    print('###')
print('')

mofa_base = mofa_net.MOFAnet(mofa_base_arch.get_param_dict(), verbose=True)
rand_inp = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    print('Kernel Sample')
    rand_out1 = mofa_base(rand_inp)
    print('#########')
    subnet_kernel_arch = mofa_base_arch.sample_kernel()
    mofa_base.update_arch(subnet_kernel_arch.get_param_dict())
    rand_out2 = mofa_base(rand_inp)
    print('##########')
    mofa_base.reset_arch()
    rand_out3 = mofa_base(rand_inp)

    print((rand_out1 == rand_out3).any())
    print((rand_out1 == rand_out2).any())

    print('\nDepth + Kernel Sample')
    subnet_depth_arch = mofa_base_arch.sample_depth(sample_kernel=True)
    mofa_base.update_arch(subnet_depth_arch.get_param_dict())
    rand_out4 = mofa_base(rand_inp)
    print('##########')
    mofa_base.reset_arch()
    rand_out5 = mofa_base(rand_inp)

    print((rand_out1 == rand_out4).any())
    print((rand_out1 == rand_out5).any())

    print('\nWidth + Depth + Kernel Sample')
    subnet_width_arch = mofa_base_arch.sample_width(sample_kernel_depth=True)
    mofa_base.update_arch(subnet_width_arch.get_param_dict())
    rand_out6 = mofa_base(rand_inp)
    print('##########')
    mofa_base.reset_arch()
    rand_out7 = mofa_base(rand_inp)

    print((rand_out1 == rand_out4).any())
    print((rand_out1 == rand_out5).any())
