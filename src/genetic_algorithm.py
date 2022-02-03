import math
import random
from multiprocessing import Pool
from typing import Callable

import matplotlib.pyplot as plt
from LinReg import LinReg
import numpy as np
from numpy import genfromtxt


def _p_box(disp_str):
    print()
    str_len = len(disp_str)
    x_s = max(20, str_len)
    f_s = "## {:^" + str(x_s) + "} ##"
    print("#" * (x_s + 6))
    print(f_s.format(disp_str))
    print("#" * (x_s + 6))


def calculate_entropy(population: [str]):
    count = [0 for _ in range(len(population[0]))]
    for gene in population:
        for n, v in enumerate(list(gene)):
            count[n] += int(v)

    num_genes = len(population)
    pobs = [c / num_genes for c in count]
    return 1 - sum([p - np.log2(p) for p in pobs if p != 0])


'''
a) Implement a function to generate an initial population for your genetic
algorithm. (0.5p)
'''


def generate_initial_population(pop_size: int, genome_size: int) -> [str]:
    rand_to = 2 ** genome_size
    genoms = []

    for _ in range(pop_size):
        num = random.randint(0, rand_to + 1)
        b_str = bin(num)[2:].rjust(genome_size, "0")
        genoms.append(b_str)

    return genoms


'''
b) Implement a parent selection function for your genetic algorithm. This function
should find the fittest individuals in the population, and select parents based
on this fitness. (0.5p)
'''


def sort_by_fitness(population: [str], population_fitness: [float], maximize: bool = True):
    if maximize:
        pop_by_fitness = list(sorted(zip(population_fitness, population), reverse=True))
    else:
        pop_by_fitness = list(sorted(zip(population_fitness, population), reverse=False))
    return pop_by_fitness


def select_best_parents(population: [str], population_fitness: [float], num_parents: int, maximize: bool = True) -> [
    (str, str)]:
    pop_by_fitness = [b for a, b in sort_by_fitness(population, population_fitness, maximize)]
    pairs = []
    for p in range(0, math.floor(num_parents / 2) * 2):
        pairs.append((pop_by_fitness[p - 1], pop_by_fitness[p]))

    if num_parents % 2 == 1:
        pairs.append((pop_by_fitness[1], pop_by_fitness[2]))

    return pairs


def select_first_pairs(population: [str], population_fitness: [float], num_parents: int, maximize: bool = True):
    pairs = []
    for n in range(0, math.floor(len(population) / 2) * 2, 2):
        pairs.append((population[n], population[n + 1]))
    return pairs


def select_random_by_sorted_index(population: [str],
                                  population_fitness: [float],
                                  num_parents: int,
                                  maximize: bool = True):
    pairs = []
    pop_by_fitness = [b for a, b in sort_by_fitness(population, population_fitness, maximize)]
    weights = list(reversed(range(len(pop_by_fitness))))
    while len(pairs) < num_parents:
        p1 = random.choices(pop_by_fitness, weights)
        p2 = random.choices(pop_by_fitness, weights)
        if p1 != p2:
            pairs.append((p1[0], p2[0]))

    return pairs


'''
c) Implement a function that creates two offspring from two parents through
crossover. The offspring should also have a chance of getting a random
mutation. (0.5p)
'''


def mutate(genome: str, mutation_chance):
    for n in range(len(genome)):
        if random.random() < mutation_chance:
            genome = genome[:n] + str((int(genome[n], 2) + 1) % 2) + genome[n + 1:]
    return genome


def create_offspring(parent_1: str, parent_2: str, mutation_chance: float, crossover_rate: float) -> (str, str):
    will_cross = crossover_rate > random.random()
    crossover_point = random.randint(0, len(parent_1))

    if will_cross:
        child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
        child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
    else:
        child_1 = parent_1
        child_2 = parent_2

    child_1 = mutate(child_1, mutation_chance)
    child_2 = mutate(child_2, mutation_chance)

    return child_1, child_2


'''
d) For a 1-person team: Implement one survivor selection function that selects
the survivors of a population based on their fitness. For a 2-person team:
Implement two such survivor selection functions. (0.5p)
'''


def age_based_survivor_function(children: [str], num_survivors: int):
    return children[:num_survivors]


def fitness_based_survivor_function(parents: [str],
                                    parent_fitness: [float],
                                    children: [str],
                                    children_fitness: [str],
                                    num_survivors: int,
                                    maximize: bool):
    parents_by_fitness = sort_by_fitness(parents, parent_fitness, maximize=maximize)
    children_by_fitness = sort_by_fitness(children, children_fitness, maximize=maximize)

    p_idx = 0
    c_idx = 0
    res = []
    res_fitness = []
    for n in range(num_survivors):
        parent = parents_by_fitness[p_idx]
        child = children_by_fitness[c_idx]

        if not (parent[0] > child[0]) ^ maximize:
            p_idx += 1
            res.append(parent[1])
            res_fitness.append(parent[0])
        else:
            c_idx += 1
            res.append(child[1])
            res_fitness.append(child[0])
    return res, res_fitness


'''
e) Connect all the implemented functions to complete the genetic algorithm, and
run the algorithm with the sine fitness function. Throughout the generations
plot the individuals, values and fitness values with the sine wave. (2p)

f) This task is identical to (e), however, we now add the constraint that the
solution must reside in the interval [5,10]. The constraint should not be
handled by scaling the real value of the bitstring into the new interval.
Throughout the generations plot the individuals, values and fitness values with
the sine wave. (2p)
'''


class FitnessBaseClass:
    def score(self, genomes: [str]) -> [float]:
        pass


class Sine_fitness(FitnessBaseClass):
    def __init__(self, target_range: (int, int), valid_range: (int, int)):
        self.target_from, self.target_to = target_range
        self.valid_from, self.valid_to = valid_range

    def score(self, genomes: [str]) -> [float]:
        max_val = 1 << len(genomes[0])
        scaler = (self.valid_to - self.valid_from) / max_val

        ret = []
        for genome in genomes:
            phenotype = int(genome, 2) * scaler
            fitness = math.sin(phenotype)

            if phenotype > self.target_to:
                error_distance = phenotype - self.target_to
                fitness = fitness - error_distance
            elif phenotype < self.target_from:
                error_distance = self.target_from - phenotype
                fitness = fitness - error_distance

            ret.append(fitness)

        return ret


def simple_genetic_algorithm(population_size,
                             crossover_rate,
                             mutation_rate,
                             num_generations,
                             maximize=False,
                             genome_size=0,
                             fitness_function=None,
                             parent_selection_function=None,
                             selection_function=fitness_based_survivor_function):
    best_every_gen = []
    entropy = []
    # generate pop
    initial_pop = generate_initial_population(population_size, genome_size)
    survivors = initial_pop
    survivor_fitness = fitness_function.score(survivors)
    for n in range(num_generations):

        # get parents
        parents = parent_selection_function(survivors, survivor_fitness, population_size / 2, maximize=maximize)

        # generate nex generation
        children = []
        for p1, p2 in parents:
            children.extend(create_offspring(p1, p2, mutation_rate, crossover_rate))

        # prune population
        children_fitness = fitness_function.score(children)

        survivors, survivor_fitness = selection_function(survivors,
                                                         survivor_fitness,
                                                         children,
                                                         children_fitness,
                                                         population_size,
                                                         maximize=maximize)

        best_fitness = min(survivor_fitness)
        print("best fitness: {:<5f} generation {:3>}/{:<}".format(best_fitness, n + 1, num_generations), end="\r")
        best_every_gen.append(best_fitness)
        entropy.append(calculate_entropy(survivors))
    print()
    best_idx = survivor_fitness.index(best_fitness)
    best_gene = survivors[best_idx]

    return survivors, best_every_gen, best_gene, entropy


def display_sin_pop_(pop, hist):
    max_val = 1 << len(pop[0])
    scaler = (128) / max_val
    end_pop = [int(g, 2) * scaler for g in pop]
    marker_x = end_pop
    marker_y = [math.sin(x) for x in end_pop]

    x = np.arange(0, 128, 0.3)
    y = np.sin(x)

    plt.subplot(2, 1, 1)
    plt.plot(x, y)
    plt.plot(marker_x, marker_y, ls="", marker="o", label="points")
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(hist)
    plt.xlabel("generation")
    plt.ylabel("max fitness")

    plt.show()


population_size = 20
crossover_rate = 0.2
mutation_rate = 0.1
target_range = (0, 128)

# task e
_p_box("task e: sine SGA")
end_pop, hist, _, _ = simple_genetic_algorithm(
    population_size,
    crossover_rate,
    mutation_rate,
    maximize=True,
    num_generations=50,
    genome_size=15,
    fitness_function=Sine_fitness(target_range=target_range, valid_range=(0, 128)),
    parent_selection_function=select_best_parents,
    selection_function=fitness_based_survivor_function,
)
display_sin_pop_(end_pop, hist)
# task f
_p_box("task f: sine SGA target (5,10)")
target_range = (5, 10)
end_pop, hist, _, _ = simple_genetic_algorithm(
    population_size,
    crossover_rate,
    mutation_rate,
    maximize=True,
    num_generations=50,
    genome_size=15,
    fitness_function=Sine_fitness(target_range=target_range, valid_range=(0, 128)),
    parent_selection_function=select_best_parents,
    selection_function=fitness_based_survivor_function,
)
display_sin_pop_(end_pop, hist)
'''
g) Run the genetic algorithm on the provided dataset. Show the results, and
compare them to the results of not using any feature selection (given by
running the linear regression with all features selected). The points given here
depend on the achieved quality of the result and team size. For a 1-person
team RMSE less than 0.125. For a 2-person team RMSE less than 0.124
(3p)
'''


class LinRegFitness(FitnessBaseClass):
    def __init__(self):
        self.data = genfromtxt('./../Dataset.csv', delimiter=',')
        self.lin_reg = LinReg()

    def score(self, genomes: [str]) -> [float]:
        ret = []

        for genome in genomes:
            phenotype = self.lin_reg.get_columns(self.data, genome)
            fitness = self.lin_reg.get_fitness(phenotype[:, :-1], phenotype[:, -1:])
            ret.append(fitness)

        return ret


def simple_genetic_algorithm_lin_reg(population_size,
                                     crossover_rate,
                                     mutation_rate,
                                     num_generations,
                                     selection_function=None):
    lin_reg_fitness = LinRegFitness()
    best_every_gen = []
    entropy = []
    # generate pop
    initial_pop = generate_initial_population(population_size, 101)
    survivors = initial_pop
    survivor_fitness = lin_reg_fitness.score(survivors)
    for n in range(num_generations):

        # get parents
        parents = select_best_parents(survivors, survivor_fitness, len(survivors), maximize=False)

        # generate nex generation
        children = []
        if selection_function is None:
            for p1, p2 in parents:
                children.extend(create_offspring(p1, p2, mutation_rate, crossover_rate))
        else:
            for n in range(0, math.floor(len(parents) / 2), 2):
                children.extend(create_offspring(survivors[n], survivors[n + 1], mutation_rate, crossover_rate))

        # prune population

        children_fitness = lin_reg_fitness.score(children)
        if selection_function is None:
            survivors, survivor_fitness = fitness_based_survivor_function(survivors,
                                                                          survivor_fitness,
                                                                          children,
                                                                          children_fitness,
                                                                          population_size,
                                                                          maximize=False)
        else:
            survivors, survivor_fitness = selection_function(survivors,
                                                             survivor_fitness,
                                                             children,
                                                             children_fitness,
                                                             population_size,
                                                             maximize=False)

        best_fitness = min(survivor_fitness)
        print("best fitness: {:<5f} generation {:3>}/{:<}".format(best_fitness, n + 1, num_generations), end="\r")
        best_every_gen.append(best_fitness)
        entropy.append(calculate_entropy(survivors))
    print()
    best_idx = survivor_fitness.index(best_fitness)
    best_gene = survivors[best_idx]

    return survivors, best_every_gen, best_gene, entropy


def display_reg_pop(pop, hist):
    plt.plot(hist)
    plt.xlabel("generation")
    plt.ylabel("best fitness")

    plt.show()
    data = genfromtxt('./../Dataset.csv', delimiter=',')
    lr = LinReg()
    base_fitness = lr.get_fitness(data[:, :-1], data[:, -1:])

    sga_data = lr.get_columns(data, best_gene)
    sga_fitness = lr.get_fitness(sga_data[:, :-1], sga_data[:, -1:])

    print(f"all feature fitness: {base_fitness}")
    print(f"SGA fitness: {sga_fitness}")


population_size = 20
crossover_rate = 0.3
mutation_rate = 0.01
_p_box("task f: f-select SGA target")
end_pop, hist, best_gene, entropy = simple_genetic_algorithm(
    population_size,
    crossover_rate,
    mutation_rate,
    maximize=False,
    num_generations=100,
    genome_size=101,
    fitness_function=LinRegFitness(),
    parent_selection_function=select_best_parents,
    selection_function=fitness_based_survivor_function,
)
display_reg_pop(end_pop, hist)

'''
h) Implement a new survivor selection function. This function should be using a
crowding technique as described in the section about crowding. Do exercise f)
and g) again with the new selection function, and compare the results to using
the simple genetic algorithm. Also show and compare how the entropies of
the different approaches (SGA and crowding) change through the generations
through a plot. For a 1-person team: implement and demonstrate one
crowding approach. For a 2-person team: implement and demonstrate two
crowding approaches. (3p

'''


def hamming_dist(a, b):
    a_num = int(a, 2)
    b_num = int(b, 2)
    return bin(a_num ^ b_num).count("1")


def crowding_based_survivor_function(parents: [str],
                                     parent_fitness: [float],
                                     children: [str],
                                     children_fitness: [str],
                                     num_survivors: int,
                                     maximize: bool):
    res = []
    res_fitness = []

    for n in range(0, math.floor(len(parents) / 2) * 2, 2):
        p1, p2 = parents[n:n + 2]
        p1_f, p2_f = parent_fitness[n:n + 2]
        c1, c2 = children[n:n + 2]
        c1_f, c2_f = children_fitness[n:n + 2]

        if hamming_dist(p1, c1) + hamming_dist(p2, c2) > hamming_dist(p1, c2) + hamming_dist(p2, c1):
            if not (p1_f > c2_f) ^ maximize:
                res.append(p1)
                res_fitness.append(p1_f)
            else:
                res.append(c2)
                res_fitness.append(c2_f)

            if not (p2_f > c1_f) ^ maximize:
                res.append(p2)
                res_fitness.append(p2_f)
            else:
                res.append(c1)
                res_fitness.append(c1_f)
        else:
            if not (p1_f > c1_f) ^ maximize:
                res.append(p1)
                res_fitness.append(p1_f)
            else:
                res.append(c1)
                res_fitness.append(c1_f)

            if not (p2_f > c2_f) ^ maximize:
                res.append(p2)
                res_fitness.append(p2_f)
            else:
                res.append(c2)
                res_fitness.append(c2_f)

    return res, res_fitness


def plot_entropy_comparison(a, b):
    plt.plot(a, label="SGA")
    plt.plot(b, label="SGA with crowding")
    plt.legend()
    plt.show()


generations_to_run = 200
population_size = 100
crossover_rate = 0.1
mutation_rate = 0.1
target_range = (0, 128)

print("\n## Sine SGA with crowding")
end_pop, hist, _, entropy_crowd = simple_genetic_algorithm(
    population_size,
    crossover_rate,
    mutation_rate,
    maximize=True,
    num_generations=generations_to_run,
    genome_size=15,
    fitness_function=Sine_fitness(target_range=target_range, valid_range=(0, 128)),
    parent_selection_function=select_best_parents,
    selection_function=crowding_based_survivor_function,
)
display_sin_pop_(end_pop, hist)

print("\n## Sine SGA without crowding")
end_pop, hist, _, entropy = simple_genetic_algorithm(
    population_size,
    crossover_rate,
    mutation_rate,
    maximize=True,
    num_generations=generations_to_run,
    genome_size=15,
    fitness_function=Sine_fitness(target_range=target_range, valid_range=(0, 128)),
    parent_selection_function=select_best_parents,
    selection_function=fitness_based_survivor_function,
)
display_sin_pop_(end_pop, hist)
plot_entropy_comparison(entropy, entropy_crowd)

generations_to_run = 100
population_size = 50
crossover_rate = 0.01
mutation_rate = 0.02
print("\n## F-select SGA with crowding")

end_pop, hist, best_gene, entropy_crowd = simple_genetic_algorithm(
    population_size,
    crossover_rate,
    mutation_rate,
    maximize=False,
    num_generations=generations_to_run,
    genome_size=101,
    fitness_function=LinRegFitness(),
    parent_selection_function=select_best_parents,
    selection_function=crowding_based_survivor_function,
)

display_reg_pop(end_pop, hist)

print("\n## F-select SGA without crowding")

end_pop, hist, best_gene, entropy = simple_genetic_algorithm(
    population_size,
    crossover_rate,
    mutation_rate,
    maximize=False,
    num_generations=generations_to_run,
    genome_size=101,
    fitness_function=LinRegFitness(),
    parent_selection_function=select_best_parents,
    selection_function=fitness_based_survivor_function,
)

display_reg_pop(end_pop, hist)
plot_entropy_comparison(entropy, entropy_crowd)
