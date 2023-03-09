from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow_tts
from tacotron2.tacotron_dataset import CharactorMelDataset
from tensorflow_tts.configs.tacotron2 import Tacotron2Config
from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.optimizers import AdamWeightDecay, WarmUp
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy
from tacotron2.trainer import Tacotron2Trainer

import redis
import numpy as np, tensorflow as tf
import uuid
import yaml
import os
import pynvml
import sys
import os
import psutil
sys.path.append(".")

import argparse
import logging
import os

import numpy as np
import yaml
from tqdm import tqdm


import json
from Genome import Genome
# declaringa a class
class obj:
      
    # constructor
    def __init__(self, dict1):
        self.__dict__.update(dict1)
   
def dict2obj(dict1):
      
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dict1), object_hook=obj)

# A creature describes a neural network
# This class creates a creature from a genome which describes the structure of the neural network
# the input and output layers are fixed. And the genomes specifies the arrangment of the hidden layers. 
from utils import interpolate
class Creature():
    def __init__(self,genome):
        self.genome = genome
        self.id = uuid.uuid1()
        self.config_obj = None
    def __build_config_from_genome(self,config_path):
        with open(config_path) as f:
            config = yaml.load(f,Loader=yaml.Loader)
            self.config_obj = config
        genome_spec = Genome.get_gene_spec()

        learning_rate_weight = self.genome.genes[0][0]

        gene = self.genome.genes[0]
        # config["optimizer_params"]["initial_learning_rate"]= float(genome_spec['initial_learning_rate']['scale'] * learning_rate_weight)
        config["optimizer_params"]["initial_learning_rate"] =  self._get_cofing_entry_from_spec(gene[0],genome_spec,'initial_learning_rate')/10
        # config["batch_size"] = self._get_cofing_entry_from_spec(gene[1],genome_spec,'batch_size')
        config["hop_size"] = self._get_cofing_entry_from_spec(gene[2],genome_spec,'hop_size')

        gene_index = 3
        for spec_name in genome_spec.keys():
            if spec_name in config['tacotron2_params']:
                config['tacotron2_params'][spec_name] = self._get_cofing_entry_from_spec(gene[gene_index],genome_spec,spec_name)
                gene_index+=1
        # config['tacotron2_params']["embedding_hidden_size"] = self._get_cofing_entry_from_spec(gene[3],genome_spec,'embedding_hidden_size')
        # config["tacotron2_params"]['initializer_range'] = self._get_cofing_entry_from_spec(gene[4],genome_spec,'initializer_range')
        # config["tacotron2_params"]['embedding_dropout_prob'] = self._get_cofing_entry_from_spec(gene[5],genome_spec,'embedding_dropout_prob')
        # config["tacotron2_params"]['n_conv_encoder'] = self._get_cofing_entry_from_spec(gene[6],genome_spec,'n_conv_encoder')
        # config["tacotron2_params"]['encoder_conv_filters'] = self._get_cofing_entry_from_spec(gene[7],genome_spec,'encoder_conv_filters')
        # config["tacotron2_params"]['encoder_conv_kernel_sizes'] = self._get_cofing_entry_from_spec(gene[8],genome_spec,'encoder_conv_kernel_sizes')
        # config["tacotron2_params"]['encoder_conv_dropout_rate'] = self._get_cofing_entry_from_spec(gene[9],genome_spec,'encoder_conv_dropout_rate')
        # config["tacotron2_params"]['encoder_lstm_units'] = self._get_cofing_entry_from_spec(gene[10],genome_spec,'encoder_lstm_units')
        # config["tacotron2_params"]['n_prenet_layers'] = self._get_cofing_entry_from_spec(gene[11],genome_spec,'prenet_units')
        # config["tacotron2_params"]['prenet_units'] = self._get_cofing_entry_from_spec(gene[12],genome_spec,'prenet_units')
        # config["tacotron2_params"]['prenet_dropout_rate'] = self._get_cofing_entry_from_spec(gene[13],genome_spec,'prenet_dropout_rate')
        # config["tacotron2_params"]['n_lstm_decoder'] = self._get_cofing_entry_from_spec(gene[14],genome_spec,'n_lstm_decoder')
        # config["tacotron2_params"]['reduction_factor'] = self._get_cofing_entry_from_spec(gene[15],genome_spec,'reduction_factor')
        # config["tacotron2_params"]['decoder_lstm_units'] = self._get_cofing_entry_from_spec(gene[16],genome_spec,'decoder_lstm_units')
        # config["tacotron2_params"]['attention_dim'] = self._get_cofing_entry_from_spec(gene[0],genome_spec,'attention_dim')
        # config["tacotron2_params"]['attention_filters'] = self._get_cofing_entry_from_spec(gene[0],genome_spec,'attention_filters')
        # config["tacotron2_params"]['attention_kernel'] = self._get_cofing_entry_from_spec(gene[0],genome_spec,'attention_kernel')
        # config["tacotron2_params"]['n_mels'] = self._get_cofing_entry_from_spec(gene[0]genome_spec,'n_mels')
        # config["tacotron2_params"]['n_conv_postnet'] = self._get_cofing_entry_from_spec(gene[0],genome_spec,'n_conv_postnet')
        # config["tacotron2_params"]['postnet_conv_filters'] = self._get_cofing_entry_from_spec(gene[0],genome_spec,'postnet_conv_filters')
        # config["tacotron2_params"]['postnet_conv_kernel_sizes'] = self._get_cofing_entry_from_spec(gene[0],genome_spec,'postnet_conv_kernel_sizes')
        # config["tacotron2_params"]['postnet_dropout_rate'] = self._get_cofing_entry_from_spec(gene[0],genome_spec,'postnet_dropout_rate')
        # config["tacotron2_params"]['prenet_units'] = self._get_cofing_entry_from_spec(gene[0],genome_spec,'prenet_units')

        print("learning_rate:",config["optimizer_params"]["initial_learning_rate"])
        with open(config_path,'w') as f:
            yaml.dump(config,f)
    def _get_cofing_entry_from_spec(self,value,gene_dict,spec_name):
        out_value = interpolate(
            value,0,1,gene_dict[spec_name]['min'],
            gene_dict[spec_name]['max'])
        if gene_dict[spec_name]['dtype'] == 'INT':
            return int(out_value)
        return float(out_value)
    def develop(self,args_dict):

        args = dict2obj(args_dict)
        print("args")
        print(args.mixed_precision)
        print(args.verbose)
        print(args.use_fal)
        print(args.train_dir)
        print(args.dev_dir)

        self.__build_config_from_genome(args.config)

        # return strategy
        STRATEGY = return_strategy()

        # set mixed precision config
        if args.mixed_precision == 1:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

        args.mixed_precision = bool(args.mixed_precision)
        args.verbose = bool(args.use_norm)
        args.use_fal = bool(args.use_fal)

        # set logger
        if args.verbose > 1:
            logging.basicConfig(
                level=logging.DEBUG,
                stream=sys.stdout,
                format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
        elif args.verbose > 0:
            logging.basicConfig(
                level=logging.INFO,
                stream=sys.stdout,
                format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
        else:
            logging.basicConfig(
                level=logging.WARN,
                stream=sys.stdout,
                format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            )
            logging.warning("Skip DEBUG/INFO messages")

        # check directory existence
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        # check arguments
        if args.train_dir is None:
            raise ValueError("Please specify --train-dir")
        if args.dev_dir is None:
            raise ValueError("Please specify --valid-dir")

        # load and save config
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config.update(vars(args))
        config["version"] = tensorflow_tts.__version__

        # get dataset
        if config["remove_short_samples"]:
            mel_length_threshold = config["mel_length_threshold"]
        else:
            mel_length_threshold = 0

        if config["format"] == "npy":
            charactor_query = "*-ids.npy"
            mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"
            align_query = "*-alignment.npy" if args.use_fal is True else ""
            charactor_load_fn = np.load
            mel_load_fn = np.load
        else:
            raise ValueError("Only npy are supported.")

        train_dataset = CharactorMelDataset(
            dataset=config["tacotron2_params"]["dataset"],
            root_dir=args.train_dir,
            charactor_query=charactor_query,
            mel_query=mel_query,
            charactor_load_fn=charactor_load_fn,
            mel_load_fn=mel_load_fn,
            mel_length_threshold=mel_length_threshold,
            reduction_factor=config["tacotron2_params"]["reduction_factor"],
            use_fixed_shapes=config["use_fixed_shapes"],
            align_query=align_query,
        )

        # update max_mel_length and max_char_length to config
        config.update({"max_mel_length": int(train_dataset.max_mel_length)})
        config.update({"max_char_length": int(train_dataset.max_char_length)})
        
        with open(os.path.join(args.outdir, "config.yml"), "w") as f:
            yaml.dump(config, f, Dumper=yaml.Dumper)
        
        for key, value in config.items():
            logging.info(f"{key} = {value}")

        train_dataset = train_dataset.create(
            is_shuffle=config["is_shuffle"],
            allow_cache=config["allow_cache"],
            batch_size=config["batch_size"]
            * STRATEGY.num_replicas_in_sync
            * config["gradient_accumulation_steps"],
        )

        valid_dataset = CharactorMelDataset(
            dataset=config["tacotron2_params"]["dataset"],
            root_dir=args.dev_dir,
            charactor_query=charactor_query,
            mel_query=mel_query,
            charactor_load_fn=charactor_load_fn,
            mel_load_fn=mel_load_fn,
            mel_length_threshold=mel_length_threshold,
            reduction_factor=config["tacotron2_params"]["reduction_factor"],
            use_fixed_shapes=False,  # don't need apply fixed shape for evaluation.
            align_query=align_query,
        ).create(
            is_shuffle=config["is_shuffle"],
            allow_cache=config["allow_cache"],
            batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
        )

        # define trainer
        trainer = Tacotron2Trainer(
            config=config,
            strategy=STRATEGY,
            steps=0,
            epochs=0,
            is_mixed_precision=args.mixed_precision,
            creature_id=str(self.id)
        )

        with STRATEGY.scope():
            # define model.
            tacotron_config = Tacotron2Config(**config["tacotron2_params"])
            tacotron2 = TFTacotron2(config=tacotron_config, name="tacotron2")
            tacotron2._build()
            tacotron2.summary()

            if len(args.pretrained) > 1:
                tacotron2.load_weights(args.pretrained, by_name=True, skip_mismatch=True)
                logging.info(
                    f"Successfully loaded pretrained weight from {args.pretrained}."
                )

            # AdamW for tacotron2
            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
                decay_steps=config["optimizer_params"]["decay_steps"],
                end_learning_rate=config["optimizer_params"]["end_learning_rate"],
            )

            learning_rate_fn = WarmUp(
                initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
                decay_schedule_fn=learning_rate_fn,
                warmup_steps=int(
                    config["train_max_steps"]
                    * config["optimizer_params"]["warmup_proportion"]
                ),
            )

            optimizer = AdamWeightDecay(
                learning_rate=learning_rate_fn,
                weight_decay_rate=config["optimizer_params"]["weight_decay"],
                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
            )

            _ = optimizer.iterations
        
        # compile trainer
        trainer.compile(model=tacotron2, optimizer=optimizer)

        # start training
        try:
            trainer.fit(
                train_dataset,
                valid_dataset,
                saved_path=os.path.join(config["outdir"], "checkpoints/"),
                resume=args.resume,
            )
        except KeyboardInterrupt:
            trainer.save_checkpoint()
            logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")
            pass
    def __str__(self) -> str:
        return str(self.id)
    def get_id(self):
        return self.id

    def __get_config_saved_path(self):
        return os.path.join(self.config_saved_dir,str(self.id),"config.yaml")
    
    def get_performance(self):
        pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
        redis_conn = redis.Redis(connection_pool=pool)
        return redis_conn.hgetall(str(self.id))
    # def __hasConfigFile(self,creature):
    #     return os.path.isfile(self.get_creature_config(creature))

    # def __get_command_to_train_creature(self):
    #     creature_out_dir = self.get_creature_out_dir(creature)
    #     creature_config_file = self.get_creature_config(creature)
    #     train_param = self.train_param + f" --out-dir {creature_out_dir} --config {creature_config_file}"
    #     return self.script_path + train_param

    # def __get_output_dir(self):
    #     return os.path.join(self.out_dir,str(self.id))
