import argparse

import yaml
from Environment import Environment
import random
from Creature import Creature
import Genome
from Reproducer import Reproducer
import numpy as np
from Population import Population
import os
import tensorflow as tf
import csv
physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

def get_fittest(performance_dict):
    max_cr = ''
    max_score = -1000
    for cr,score in performance_dict.items():
        if score > max_score:
            max_cr=cr
    return max_cr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FastSpeech (See detail in tensorflow_tts/bin/train-fastspeech.py)"
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        type=str,
        help="directory including training data. ",
    )
    parser.add_argument(
        "--dev-dir",
        default=None,
        type=str,
        help="directory including development data. ",
    )
    parser.add_argument(
        "--use-norm", default=1, type=int, help="usr norm-mels for train or raw."
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--mixed_precision",
        default=0,
        type=int,
        help="using mixed precision for generator or not.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        nargs="?",
        help="pretrained weights .h5 file to load weights from. Auto-skips non-matching layers",
    )
    parser.add_argument(
        "--use-fal",
        default=0,
        type=int,
        help="Use forced alignment guided attention loss or regular",
    )
    args = parser.parse_args()
    print(args)

    pop = Population(pop_size=10,gene_size=2,gene_length=len(Genome.Genome.get_gene_spec().keys()))

    outdir = os.path.join(os.getcwd(),args.outdir)
    dev_dir = os.path.join(os.getcwd(),args.dev_dir)
    train_dir = os.path.join(os.getcwd(),args.train_dir)
    config = os.path.join(os.getcwd(),args.config)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    num_of_generation=20
    env = Environment(population=pop,
                                default_config_path=config,
                                dev_dir=dev_dir,
                                train_dir=train_dir,
                                out_dir=os.path.join(outdir,f'gen_{i}'))
    print(pop)

    overall_performance_filepath = os.path.join(outdir,'overall_performance.csv')
    dead_creature_filepath = os.path.join(outdir,'dead_creature.csv')

    for i in range(num_of_generation):
        performance_table = {}
        env.set_out_dir(os.path.join(outdir,f'gen_{i}'))

        for c in pop.get_creatures():
            # develop creature
            survived = env.developCreature(c)
            
            if not survived:
                print('creature dead:',str(c.id))
                performance_table[str(c.id)] = 0
                record_dict={'stop_token_loss':-1,
                'mel_loss_before':-1,
                'mel_loss_after':-1,
                'guided_attention_loss':-1,
                'memory_usage':-1,
                'cpu_usage':-1,
                'gpu_usage':-1,
                'creature_id':str(c.id),
                'generation':i,
                'perforamce':0}
            else:
                # get the performance of the creature
                record_dict = c.get_performance()
                print(record_dict)

                # save the performance for reproduction
                loss = float(record_dict['stop_token_loss']) + float(record_dict['mel_loss_before'])+ float(record_dict['mel_loss_after'])+float(record_dict['guided_attention_loss'])
                performance_table[str(c.id)] = 1/(loss+1) * 100
                print(performance_table)
                # save the performance of the creature to log
                record_dict["creature_id"] = str(c.id)
                record_dict["generation"] = i
                record_dict["perforamce"] = performance_table[str(c.id)] 

            create_overall_file = False
            if not os.path.exists(overall_performance_filepath):
                create_overall_file = True

            with open(overall_performance_filepath,'a',newline="") as f:
                writer = csv.DictWriter(f,fieldnames=record_dict.keys())
                if create_overall_file:
                    writer.writeheader()
                writer.writerow(record_dict)

        log = ''

        log += "Pop Creatures:"+str(pop.get_all_creatures_id()) +'\n'
        log += "Performance Table:"+str(performance_table) +'\n'
        
        r = Reproducer(performance_table)
        print('perforamance table:')
        print(performance_table)
        # Save Elite Creature
        elite_id = r.find_max_creature()
        log += "Elite Creature:" + str(elite_id) +'\n'
        gen_dir = os.path.join(outdir,f'gen_{i}')
        with open(os.path.join(gen_dir,'elite.yaml'),'w') as f:
            yaml.dump(pop.get_creature(elite_id).config_obj,f)

        # Selection
        parent_1_id = r.select(k=2)
        parent_2_id = r.select(k=2)

        parent_1 = pop.get_creature(parent_1_id)
        parent_2 = pop.get_creature(parent_2_id)

        # Reproduction
        child_genome = r.cross_over_creature(parent_1,parent_2)

        r.point_mutate(child_genome)

        child = Creature(child_genome)

        pop.add_creature(child)
        removed_creature_id = r.find_min_creature()

        log += "New_creature_id:" + str(child.id) +'\n'
        log += "removed_creature_id:" + removed_creature_id+'\n'

        pop.delete_creature(removed_creature_id)

        log += "Pop Creatures After Remove Min Creature :"+str(pop.get_all_creatures_id()) +'\n'

        with open(os.path.join(outdir,'ga_output.txt'), 'a') as f:
            f.write("Generation:"+str(i))
            f.write(log)


# pop = Population()
# performance_dict = {}
# mae_dict={}
# for i in range(num_of_generation):
#     for cr in pop.population:
#         res = env.developCreature(cr)
#         avg_mae = [np.mean([x[i] for x in res['mae']]) for i in range(8)]
#         print(avg_mae)
#         final_score = avg_mae[-1:][0]
#         print(final_score)
#         performance_dict[str(cr)] = 1/(final_score - res['runtime']/10000)
#         mae_dict[str(cr)]=final_score
    # reproducer = Reproducer.Reproducer(performance_dict,2)
    # new_cr = reproducer.select()
    # reproducer.point_mutate(new_cr)
    # pop.add_creature(new_cr)
    # print(mae_dict)
    # print(performance_dict)
    # fittest = get_fittest(performance_dict)
    # f_cr = pop.get_creature(fittest)
    # print('fittest:',f_cr,', score: ',performance_dict[fittest],'\n','NN Spec:',f_cr.genome.genes,"MAE:",mae_dict[str(fittest)]    )

# cr = Creature.Creature(Genome.Genome(2))
# cr.develop(train_data_shape=[1024,128])
# print(cr.model.summary())

    
