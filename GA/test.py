from distutils.command.config import config
import os
import random
import unittest

from regex import D
from Genome import Genome
from Creature import Creature
from Reproducer import Reproducer
from Population import Population
from Environment import Environment
config_filepath = "./GA/conf/init_tacotron2.v1.yaml"
config_filepath = os.path.abspath(config_filepath)
class GenomeTest(unittest.TestCase):
    def test_genome_not_none(self):
        gen = Genome(3)
        self.assertIsNotNone(gen.genes)
    def test_Genome_Length(self):
        gen = Genome(3)
        self.assertEqual(len(gen.genes),3)
    # def test_invalid_genome_length_2(self):
    #     gen = Genome(2)
    #     self.assertEqual(len(gen.genes),2)
    def test_invalid_genome_length_0(self):
        with self.assertRaises(Exception):
            gen = Genome(0)
class CreatureTest(unittest.TestCase):
    def test_creature_has_genome(self):
        gen = Genome();
        gen.genes = [[0,2],[1,2]]
        creature = Creature(gen)
        self.assertIsNotNone(creature.genome)
    def test_model_created_successful(self):
        gen = Genome();
        gen.genes = [[0,2],[1,2]]
        creature = Creature(gen)
        self.assertEqual(len(creature.model.layers),0)
    def test_model_built_successful(self):
        gen = Genome();
        gen.genes = [[0,2],[1,2]]
        creature = Creature(gen)
        creature.develop(1024)
        self.assertNotEqual(len(creature.model.layers),0)
class EnvironmentTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.pop = Population(config_filepath)
        self.out_dir = "./creatures"
        self.env = Environment(population=self.pop,script_path="./examples/tacotron2/train_tacotron2.py",dev_dir="./dump_ljspeech/valid/",train_dir="./dump_ljspeech/train/",out_dir=self.out_dir)

    def test_get_command(self):
        print(self.env.get_command_to_train_creature(self.pop.get_creatures()[1]))
    def test_get_out_dir(self):
        creature = self.pop.get_creatures()[1]
        self.assertEqual(self.env.get_creature_out_dir(creature), os.path.join(self.out_dir,str(creature)))
    def test_get_creature_config(self):
        creature = self.pop.get_creatures()[1]
        self.assertEqual(self.env.get_creature_config(creature), os.path.join(self.out_dir,str(creature),"config.yaml"))

class ReproducerTest(unittest.TestCase):
    # def test_select_max_creature_successfully(self):
    #     reproducer = Reproducer({'a':1,'b':2,'c':3,'d':4})
    #     max_id = reproducer.select_max_creature_id(['a','b','d'])
    #     self.assertEqual(max_id,'d')
    # def test_select_max_creature_failed(self):
    #     reproducer = Reproducer({'a':1,'b':2,'c':3,'d':4})
    #     max_id = reproducer.select_max_creature_id(['a','b','d'])
    #     self.assertNotEqual(max_id,'a')
    # def test_select_successfully(self):
    #     reproducer = Reproducer({'a':1,'b':2,'c':3,'d':4})
    #     random.seed(10)
    #     max_id = reproducer.select()
    #     self.assertEqual(max_id,'b')
    pass
if __name__ == '__main__':
    unittest.main()