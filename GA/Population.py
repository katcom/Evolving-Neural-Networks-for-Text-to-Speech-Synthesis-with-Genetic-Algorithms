from Creature import Creature
from Genome import Genome
import yaml
class Population():
    def __init__(self,pop_size=3,gene_size=1,gene_length=10):
        # with open(default_config_filepath) as f:
        #     default_config = yaml.load(f,Loader=yaml.Loader)
        self.population = [Creature(Genome(gene_size,gene_length)) for i in range(pop_size)]
        self.pop_table = {}
        for creature in self.population:
            self.pop_table[str(creature.id)]=creature
        print(self.pop_table)
    def get_creature(self,id):
        return self.pop_table[id]
    def add_creature(self,creature):
        self.pop_table[str(creature)] = creature
        self.population.append(creature)
    def get_creatures(self):
        return list(self.pop_table.values())
    def delete_creature(self,creature_id):
        del self.pop_table[str(creature_id)]

        for i in range(len(self.population)):
            if str(self.population[i].id) == creature_id:
                del self.population[i]
                break
    def get_all_creatures_id(self):
        return [str(c.id) for c in self.population]