import torch

import mofa_net

mofa_base_arch = mofa_net.MOFA_Net_Arch.create_largest()

mofa1_arch = mofa_base_arch.mutate(0.7)
mofa2_arch = mofa_base_arch.mutate(0.5)

mofa3_arch = mofa_net.MOFA_Net_Arch.crossover(mofa1_arch, mofa2_arch) 

print('\n', mofa_base_arch.get_param_dict(), mofa_base_arch.get_num_weights())
print('\n', mofa1_arch.get_param_dict(), mofa1_arch.get_num_weights())
print('\n', mofa2_arch.get_param_dict(), mofa2_arch.get_num_weights())
print('\n', mofa3_arch.get_param_dict(), mofa3_arch.get_num_weights())

print('\n#######################\n')

mofa = mofa_net.MOFAnet(mofa_base_arch.get_param_dict())
print('\n', mofa.get_arch())
mofa.update_arch(mofa1_arch.get_param_dict())
print('\n', mofa.get_arch())
mofa.update_arch(mofa2_arch.get_param_dict())
print('\n', mofa.get_arch())
mofa.update_arch(mofa3_arch.get_param_dict())
print('\n', mofa.get_arch())
mofa.reset_arch()
print('\n', mofa.get_arch())

print('\n#######################\n')

rand_inp = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    out_base_1 = mofa(rand_inp)
    mofa.update_arch(mofa1_arch.get_param_dict())
    out1 = mofa(rand_inp)
    mofa.update_arch(mofa2_arch.get_param_dict())
    out2 = mofa(rand_inp)
    mofa.update_arch(mofa3_arch.get_param_dict())
    out3 = mofa(rand_inp)
    mofa.reset_arch()
    out_base_2 = mofa(rand_inp)

print(out_base_1)
print(out1)
print(out2)
print(out3)
print(out_base_2)

print((out_base_1 == out_base_2).all())
