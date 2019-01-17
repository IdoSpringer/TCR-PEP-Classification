import json


def set_model(arguments):
    # classifier model to use
    arguments['divide'] = arguments['test'] == 'yes'
    if bool(arguments['bi_lstm']):
        arguments['num_layers'] = int(arguments['num_layers'])
        arguments['bi_lstm'] = bool(arguments['bi_lstm'])
        arguments['model_name'] = 'upgrade'
    elif arguments['num_lstm'] == 2:
        arguments['model_name'] = 'double'
    else:
        arguments['model_name'] = 'one'


def set_dims(arguments):
    arguments['embedding_dim'] = int(arguments['embedding_dim'])
    arguments['hidden_dim'] = int(arguments['hidden_dim'])


def set_training(arguments):
    # number of classes
    num_of_peptides = arguments['num_of_peptides']
    if num_of_peptides.isdigit():
        num_of_peptides = int(num_of_peptides)
        num_of_training = range(num_of_peptides, num_of_peptides+1)
    elif num_of_peptides == 'all':
        num_of_training = range(2, 16)
    else:
        v1, v2 = int(num_of_peptides.split('-')[0]), int(num_of_peptides.split('-')[-1])
        num_of_training = range(v1, v2+1)
    arguments['num_of_training'] = num_of_training


with open("param_file.json", 'r') as param_file:
    # Load parameters from file
    arguments = json.load(param_file)
    set_model(arguments)
    set_training(arguments)
    set_dims(arguments)
