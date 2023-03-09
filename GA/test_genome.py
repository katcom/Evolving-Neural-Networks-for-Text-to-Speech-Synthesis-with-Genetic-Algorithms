import unittest
from Genome import Genome
from Reproducer import Reproducer
r = Reproducer({})
from Creature import Creature
class Genome_Test(unittest.TestCase):
    def testPointMutate(self):
        g = Genome()
        print(g.genes)
        new_gen = r.point_mutate(g.genes,rate=0.15)
        print(new_gen)
    # def testPointMutateCreature(self):
    #     g = Genome()
    #     c = Creature.Creature(g)
    #     print(c.genome.genes)
    #     new_creature = r.point_mutate_creature(c,rate=0.15)
    #     print(new_creature.genome.genes)
    def testCrossOver(self):
        g1 = Genome().genes[0]
        g2 = Genome().genes[0]
        print("g1:",g1)
        print("g2",g2)
        new_g = r.cross_over(g1,g2)
        print("new_g:",new_g)
    def testFindMinCreature(self):
        r = Reproducer({"a":1,"b":2,"c":3,"d":4,"e":5})
        self.assertEqual(r.find_min_creature(),'a')
    def testFindMaxCreature(self):
        pt = {"a":1,"b":2,"c":3,"d":4,"e":5}
        r = Reproducer(pt)
        self.assertEqual(r.find_max_creature_id(pt.keys()),'e')
    def testFindMaxCreatureFromALLCreature(self):
        pt = {"a":1,"b":2,"c":3,"d":4,"e":5}
        r = Reproducer(pt)
        self.assertEqual(r.find_max_creature(),'e')
    def testCrossoverCreature(self):
        g1 = Genome()
        g2 = Genome()

        c1 = Creature(g1)
        c2 = Creature(g2)
        new_c = r.cross_over_creature(c1,c2)
        print("g1:",g1.genes)
        print("g2:",g2.genes)
        print("new_c:",new_c.genes)
if __name__ == "__main__":
    unittest.main()