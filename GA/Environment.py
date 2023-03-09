# A population describes a group of creatures which specifies neural networks.
# An environment describes the datasets and the creatures to develop
from tensorflow.keras.datasets import boston_housing
import numpy as np, tensorflow as tf
import time
import os
import csv
import yaml
import gc
physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

# from Creature import Creature
# from Genome import Genome
class Environment():
    def __init__(self,population,default_config_path,train_dir,dev_dir,out_dir):
        self.default_config_path =default_config_path
        #self.prepare_dataset()
        self.population=population
        # self.train_param = f"   --train-dir {train_dir} \
        #     --dev-dir {dev_dir} \
        #     --use-norm 1 \
        #     --mixed_precision 0 \
        #     --resume "
        self.out_dir = out_dir
        self.train_dir = train_dir
        self.dev_dir = dev_dir

    '''
        Train neural networks
        If succeed, return true, else false.
    '''
    def developCreature(self,creature):
        # command = self.get_command_to_train_creature(creature)
        out_dir = self.get_creature_out_dir(creature)
        config_path = self.get_creature_config(creature)

        if not self.hasConfigFile(creature):
            self.createConfigFile(creature)
        # creature.genome.spec["optimizer_params"]["initial_learning_rate"] = creature.genome.genes[0]
        
        args_dict = {
            "train_dir":self.train_dir,
            "dev_dir":self.dev_dir,
            "use_norm":1,
            "outdir":self.get_creature_out_dir(creature),
            "config":self.get_creature_config(creature),
            "resume":"",
            "verbose":1,
            "mixed_precision":0,
            "pretrained":"",
            "use_fal":0
        }
        print(args_dict)
        print("Train:",creature.id)
        print("Config:",config_path)
        print("Output:",out_dir)
        self.save_creature_dna(creature)
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            creature.develop(args_dict)
            return True
        except:
            return False
        # os.system(command)

    def createConfigFile(self,creature):
        with open(self.default_config_path) as f:
            config = yaml.load(f,Loader=yaml.Loader)
        
        if not os.path.exists(self.get_creature_out_dir(creature)):
            os.mkdir(self.get_creature_out_dir(creature))
            print('create',self.get_creature_out_dir(creature))
        print(self.get_creature_out_dir(creature))
        config_path = self.get_creature_config(creature)

        with open(config_path,'w') as f:
            yaml.dump(config,f)
    def hasConfigFile(self,creature):
        return os.path.exists(self.get_creature_config(creature))
    def get_creature_out_dir(self,creature):
        return os.path.join(self.out_dir,str(creature.id))
    def get_creature_config(self,creature):
        return os.path.join(self.out_dir,str(creature.id),"config.yaml")

    def set_out_dir(self,dir):
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.out_dir=dir
    def save_creature_dna(self,creature):
        filename = os.path.join(self.get_creature_out_dir(creature),'genome.csv')
        with open(filename,'w',newline="") as f:
            writer = csv.writer(f)
            writer.writerows(creature.genome.genes)
# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train Tacotron2."""
