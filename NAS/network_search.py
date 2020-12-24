import random
import numpy as np
import mofa_net


def check_constraint(new_sample, constraint):
    if 'max_num_weights' in constraint:
        if new_sample.get_num_weights() <= constraint['max_num_weights']:
            return True
        
    return False


def calc_accuracy(child_net):
    return random.random()


def calc_efficiency(child_net):
    return 1.0


class EvolutionSearch:
    def __init__(self, population_size=100, prob_mutation=0.1, ratio_mutation=0.5, ratio_parent=0.25, num_iter=500):
        self.population_size = population_size
        self.prob_mutation = prob_mutation
        self.ratio_mutation = ratio_mutation
        self.ratio_parent = ratio_parent
        self.num_iter = num_iter
        
    def set_model_arch(self, arch):
        self.arch = arch
    
    def get_random_valid_sample(self, constraint):
        is_sample_ok = False
        while not is_sample_ok:
            sample = self.arch.mutate(1.0)
            is_sample_ok = check_constraint(sample, constraint)
            
        return sample
    
    def mutate_valid_sample(self, sample, constraint):
        is_sample_ok = False
        while not is_sample_ok:
            new_sample = sample.mutate(self.prob_mutation)
            is_sample_ok = check_constraint(new_sample, constraint)
            
        #print(f'Mutation: {new_sample.get_num_weights()}')
        return new_sample
    
    def crossover_valid_sample(self, sample1, sample2, constraint):
        is_sample_ok = False
        while not is_sample_ok:
            new_sample = mofa_net.MOFA_Net_Arch.crossover(sample1, sample2)
            is_sample_ok = check_constraint(new_sample, constraint)
            
        #print(f'Crossover: {new_sample.get_num_weights()}')
        return new_sample
    
    def run(self, constraint):
        num_mutations = int(round(self.population_size * self.ratio_mutation))
        num_parents = int(round(self.population_size * self.ratio_parent))
        
        population = []
        child_pool = []
        best_acc = -9999
        best_arch = None
        
        for _ in range(self.population_size):
            child_net = self.get_random_valid_sample(constraint)
            child_net_acc = calc_accuracy(child_net)
            child_net_eff = calc_efficiency(child_net)
            
            population.append((child_net, child_net_acc, child_net_eff))
            
        for n in range(self.num_iter):
            # Total popoulation size is equal to (population_size + num_parents) 
            # after first iteration.
            parents = sorted(population, key=lambda x: x[1], reverse=True)[:num_parents]
            
            acc = parents[0][1]
            print(f'Iteration: {n}\tMax. Accuracy: {acc}')
            if acc > best_acc:
                best_arch = parents[0][0] 
            
            population = parents
            child_pool = []
            
            for _ in range(num_mutations):
                sample = population[np.random.randint(num_parents)][0]
                child_net = self.mutate_valid_sample(sample, constraint)
                child_net_acc = calc_accuracy(child_net)
                child_net_eff = calc_efficiency(child_net)
                
                population.append((child_net, child_net_acc, child_net_eff))
                
            for _ in range(self.population_size - num_mutations):
                sample1 = population[np.random.randint(num_parents)][0]
                sample2 = population[np.random.randint(num_parents)][0]
                child_net = self.crossover_valid_sample(sample1, sample2, constraint)
                child_net_acc = calc_accuracy(child_net)
                child_net_eff = calc_efficiency(child_net)
                
                population.append((child_net, child_net_acc, child_net_eff))
                
        return best_arch, best_acc
