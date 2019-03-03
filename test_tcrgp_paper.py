import os
import csv
from random import shuffle

from pair_sampling.new_train import *


def read_all_data():
    directory = 'TCRGP/training_data'
    peps = []
    data_dict = {}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv") and filename.startswith("vdj"):
            # get peptide from filename
            pep = filename.split('_')[-1].split('.')[0]
            peps.append(pep)
            data_dict[pep] = []
            with open(directory + '/' + filename, 'r') as csv_file:
                csv_file.readline()
                csv_ = csv.reader(csv_file)
                for row in csv_:
                    tcr = row[-1]
                    sign = 'p'
                    if row[1] == 'control':
                        sign = 'n'
                    data_dict[pep].append((tcr, pep, sign, 1))
        else:
            continue
    return peps, data_dict


def split_train_test(pep, data_dict):
    train = []
    for key in data_dict:
        if not key == pep:
            train += data_dict[key]
    test = data_dict[pep]
    shuffle(train)
    shuffle(test)
    return train, test


peps, data_dict = read_all_data()
train, test = split_train_test(peps[0], data_dict)


def main(argv):
    # Word to index dictionary
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}

    # Set all parameters and program arguments
    device = argv[2]
    args = {}
    args['train_auc_file'] = argv[3]
    args['test_auc_file'] = argv[4]
    args['siamese'] = False
    params = {}
    params['lr'] = 1e-3
    params['wd'] = 0
    params['epochs'] = 200
    params['batch_size'] = 50
    params['lstm_dim'] = 30
    params['emb_dim'] = 10
    params['dropout'] = 0.1
    params['option'] = 0
    # Load data

    '''
    pairs_file = 'pairs_data/weizmann_pairs.txt'
    if argv[-1] == 'cancer':
        pairs_file = 'pairs_data/cancer_pairs.txt'
    if argv[-1] == 'shugay':
        pairs_file = 'pairs_data/shugay_pairs.txt'
    train, test = d.load_data(pairs_file)
    '''
    peps, data_dict = read_all_data()
    for pep in peps:
        pass
    train, test = split_train_test(peps[0], data_dict)

    # train
    train_tcrs, train_peps, train_signs = get_lists_from_pairs(train)
    convert_data(train_tcrs, train_peps, amino_to_ix)
    train_batches = get_batches(train_tcrs, train_peps, train_signs, params['batch_size'])

    # test
    test_tcrs, test_peps, test_signs = get_lists_from_pairs(test)
    convert_data(test_tcrs, test_peps, amino_to_ix)
    test_batches = get_batches(test_tcrs, test_peps, test_signs, params['batch_size'])

    # Train the model
    model = train_model(train_batches, test_batches, device, args, params)

    # Save trained model
    torch.save({
                'model_state_dict': model.state_dict(),
                'amino_to_ix': amino_to_ix
                }, argv[1])
    pass


if __name__ == '__main__':
    main(sys.argv)
