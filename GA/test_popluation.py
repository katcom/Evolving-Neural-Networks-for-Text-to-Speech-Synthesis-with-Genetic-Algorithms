import unittest
from Genome import Genome
from Reproducer import Reproducer
from Population import Population
from Creature import Creature
import unittest
from Genome import Genome
from Reproducer import Reproducer
r = Reproducer({})
from Creature import Creature
pop = Population()
print(pop)
class Population_Test(unittest.TestCase):
    # def testPrintCreature(self):
    #     print(pop.pop_table)

    def testDeletedCreatureNotInPopTable(self):
        cr = pop.population[0]
        print('delete:',str(cr))
        pop.delete_creature(str(cr))
        print(pop.pop_table)
        self.assertTrue(str(cr) not in pop.pop_table.keys())

    def testDeletedCreatureNotInPopulation(self):
        cr = pop.population[0]
        print('delete:',str(cr))
        pop.delete_creature(str(cr))
        print([str(p) for p in pop.population])
        self.assertTrue(cr not in pop.population)
    def testGetCreatureID(self):
        print("ALL:",pop.get_all_creatures_id())

if __name__ == "__main__":
    unittest.main()
    