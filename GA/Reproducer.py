# A reproducer is responsible for choosing parents to reproduce and performing mutation

import copy
import random
from Genome import Genome
from select import select

class Reproducer():
    def __init__(self,performance_table,seed=0):
        self.performance_table=performance_table
        random.seed(seed)
    def select(self,k=3):
        max_id = self.find_max_creature_in_random_interval(k) 
        return max_id
    def find_max_creature_in_random_interval(self,k):
        creatures = []
        keys = list(self.performance_table.keys())
        # select k creatures from the population
        creature_indices= random.sample(range(0, len(keys)), k)

        for index in creature_indices:
            creature_id =keys[index]
            creatures.append(creature_id)

        max_id = self.find_max_creature_id(creatures)
        return max_id
    def find_max_creature_id(self,creatures):
        max_creature_id = ''
        max_score =0
        for creature_id in creatures:
            if max_score <= self.performance_table[creature_id]:
                max_creature_id = creature_id
                max_score = self.performance_table[creature_id]
        return max_creature_id
    def find_max_creature(self):
        return self.find_max_creature_id(self.performance_table.keys())
    def find_min_creature(self):
        min_score = 9999999999
        for creature_id in self.performance_table.keys():
            if min_score > self.performance_table[creature_id]:
                min_creature_id = creature_id
                min_score = self.performance_table[creature_id]
        return min_creature_id
    def point_mutate_creature(self,creature,rate=0.01,amount=0.1):
        new_gen = self.point_mutate(creature.genome.genes,rate=0.01,amount=0.1)
        creature.genome.genes = new_gen
    def point_mutate(self,genome,rate=0.01,amount=0.1):
        new_genome = copy.copy(genome)
        for gene in new_genome.genes:
            for i in range(len(gene)):
                if random.random() < rate:
                    sign = random.choice([1,-1])
                    gene[i] = gene[i] + amount *sign 
        return genome
    # def cross_over_creature(self,creature_1,creature_2):
    #     creature_1,creature_2
    #     pos = random.randint(0,len(creature_1)
    #     pass
    def cross_over(self,gene_1,gene_2,crossing_rate=0.5,seed=0):
        # pos = random.randint(0,len(gene_1)-1)
        # num_of_cross_point = random.randint(0,len(gene_1))
        random.seed(seed)
        cross_pos = random.sample(range(len(gene_1)), k=int(crossing_rate*len(gene_1)))

        print(cross_pos)
        new_gene = copy.copy(gene_1)
        for i in cross_pos:
            new_gene[i] = gene_2[i]
        return new_gene
    def cross_over_creature(self,creature_1,creature_2):
        # pos = random.randint(0,len(gene_1)-1)
        new_genome = Genome(num_of_genes=len(creature_1.genome.genes))
        for i in range(len(creature_1.genome.genes)):
            new_genome.genes[i] = self.cross_over(creature_1.genome.genes[i],creature_2.genome.genes[i])
        return new_genome

# reproducer = Reproducer({'a':1,'b':2,'c':3,'d':4},2)
# reproducer.select()
