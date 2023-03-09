from ruamel.yaml import YAML
import sys
# yaml = YAML(typ='safe')
yaml = YAML()
filename  = "tacotron2.v1.yaml"
output = "tacotron2.v1.modified.yaml3"
with open(filename) as f:
    config = yaml.load(f,)
    pass


config["optimizer_params"]["initial_learning_rate"] = 4
print(config['var_train_expr'])
with open(output,'w') as f:
    yaml.dump(config,f)

with open(output,'w') as f:
    yaml.dump(config,f)
# yaml.dump(config, sys.stdout)