import mofa_net
import network_search

constraint = {'max_num_weights': 4e5}

mofa_base = mofa_net.MOFA_Net_Arch.create_largest()

evo_search = network_search.EvolutionSearch()
evo_search.set_model_arch(mofa_base)
best_arch, best_acc = evo_search.run(constraint)

print(best_arch)