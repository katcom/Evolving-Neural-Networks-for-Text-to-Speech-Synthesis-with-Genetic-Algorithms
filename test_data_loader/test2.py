import yaml

filename  = "tacotron2.v1.yaml"
output = "tacotron2.v1.modified.yaml"
with open(filename) as f:
    config = yaml.load(f,Loader=yaml.Loader)
    pass

print(config)
# config["optimizer_params"]["initial_learning_rate"] = 4
# print(config['var_train_expr'])
# with open(output+'2','w') as f:
#     yaml.dump(config,f)

# yaml.dump(config, sys.stdout)